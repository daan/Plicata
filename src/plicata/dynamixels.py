"""
dynamixels.py -- thin wrapper around dynamixel_sdk for X-series servos (XL430-W250).

Based on the bus/servo classes from an earlier project. Positions are raw
encoder ticks (4096 per turn); angles are degrees.
"""

from dynamixel_sdk import COMM_SUCCESS, GroupSyncWrite, PacketHandler, PortHandler
from serial_device import get_ports_with_vid

# X-series control table (XL430-W250; XM430 uses the same addresses)
ADDR_DRIVE_MODE = 10
ADDR_OPERATING_MODE = 11
ADDR_HOMING_OFFSET = 20
ADDR_PWM_LIMIT = 36
ADDR_TORQUE_ENABLE = 64
ADDR_HARDWARE_ERROR = 70
ADDR_GOAL_PWM = 100
ADDR_PROFILE_ACCELERATION = 108
ADDR_PROFILE_VELOCITY = 112
ADDR_GOAL_POSITION = 116
ADDR_MOVING = 122
ADDR_PRESENT_LOAD = 126        # XL430: load in 0.1 %
ADDR_PRESENT_POSITION = 132
ADDR_PRESENT_VOLTAGE = 144
ADDR_PRESENT_TEMPERATURE = 146

MODE_POSITION = 3
MODE_EXTENDED_POSITION = 4     # multi-turn; turn count is lost at power-off

TORQUE_ENABLE = 1
TORQUE_DISABLE = 0
PWM_MAX = 885

PROTOCOL_VERSION = 2.0
BAUDRATE = 57600               # Dynamixel default
FTDI_VID = 1027                # U2D2

TICKS_PER_TURN = 4096


def signed(value, nbytes):
    bits = 8 * nbytes
    return value - (1 << bits) if value >= 1 << (bits - 1) else value


class DynamixelBus:
    def __init__(self, device=None, baudrate=BAUDRATE):
        if device is None:
            ports = get_ports_with_vid(FTDI_VID)
            if not ports:
                raise SystemExit("no U2D2 (FTDI) serial port found; is it plugged in and Dynamixel Wizard closed?")
            device = ports[-1]
        self.DEVICENAME = device
        self.portHandler = PortHandler(device)
        self.packetHandler = PacketHandler(PROTOCOL_VERSION)
        if not self.portHandler.openPort():
            raise SystemExit(f"failed to open {device} (Dynamixel Wizard still connected?)")
        if not self.portHandler.setBaudRate(baudrate):
            raise SystemExit(f"failed to set baudrate {baudrate}")

    def close_port(self):
        self.portHandler.closePort()

    def _check(self, dxl_comm_result, dxl_error, what):
        if dxl_comm_result != COMM_SUCCESS:
            raise IOError(f"{what}: {self.packetHandler.getTxRxResult(dxl_comm_result)}")
        if dxl_error != 0:
            print(f"{what}: {self.packetHandler.getRxPacketError(dxl_error)}")

    def ping(self, dxl_id):
        model, dxl_comm_result, dxl_error = self.packetHandler.ping(self.portHandler, dxl_id)
        return model if dxl_comm_result == COMM_SUCCESS else None

    def write1(self, dxl_id, address, value):
        self._check(*self.packetHandler.write1ByteTxRx(self.portHandler, dxl_id, address, value),
                    f"id {dxl_id} write @{address}")

    def write2(self, dxl_id, address, value):
        self._check(*self.packetHandler.write2ByteTxRx(self.portHandler, dxl_id, address, value & 0xFFFF),
                    f"id {dxl_id} write @{address}")

    def write4(self, dxl_id, address, value):
        self._check(*self.packetHandler.write4ByteTxRx(self.portHandler, dxl_id, address, value & 0xFFFFFFFF),
                    f"id {dxl_id} write @{address}")

    def read1(self, dxl_id, address):
        value, *res = self.packetHandler.read1ByteTxRx(self.portHandler, dxl_id, address)
        self._check(*res, f"id {dxl_id} read @{address}")
        return value

    def read2(self, dxl_id, address):
        value, *res = self.packetHandler.read2ByteTxRx(self.portHandler, dxl_id, address)
        self._check(*res, f"id {dxl_id} read @{address}")
        return signed(value, 2)

    def read4(self, dxl_id, address):
        value, *res = self.packetHandler.read4ByteTxRx(self.portHandler, dxl_id, address)
        self._check(*res, f"id {dxl_id} read @{address}")
        return signed(value, 4)

    def sync_goal_positions(self, goals):
        """goals: {id: ticks}. Sends all goals in one packet so the servos start together."""
        gsw = GroupSyncWrite(self.portHandler, self.packetHandler, ADDR_GOAL_POSITION, 4)
        for dxl_id, ticks in goals.items():
            gsw.addParam(dxl_id, list((int(ticks) & 0xFFFFFFFF).to_bytes(4, "little")))
        res = gsw.txPacket()
        if res != COMM_SUCCESS:
            raise IOError(f"sync write: {self.packetHandler.getTxRxResult(res)}")


class DynamixelServo:
    def __init__(self, bus, id, name=""):
        self.bus = bus
        self.id = id
        self.name = name or f"id {id}"

    def enable_torque(self):
        self.bus.write1(self.id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)

    def disable_torque(self):
        self.bus.write1(self.id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)

    def torque_enabled(self):
        return self.bus.read1(self.id, ADDR_TORQUE_ENABLE) == TORQUE_ENABLE

    def operating_mode(self):
        return self.bus.read1(self.id, ADDR_OPERATING_MODE)

    def set_operating_mode(self, mode):
        """EEPROM: torque must be off."""
        self.bus.write1(self.id, ADDR_OPERATING_MODE, mode)

    def homing_offset(self):
        return self.bus.read4(self.id, ADDR_HOMING_OFFSET)

    def set_homing_offset(self, ticks):
        """EEPROM: torque must be off. Present Position = raw position + homing offset."""
        self.bus.write4(self.id, ADDR_HOMING_OFFSET, ticks)

    def hardware_error(self):
        return self.bus.read1(self.id, ADDR_HARDWARE_ERROR)

    def present_position(self):
        return self.bus.read4(self.id, ADDR_PRESENT_POSITION)

    def goal_position(self):
        return self.bus.read4(self.id, ADDR_GOAL_POSITION)

    def set_goal_position(self, goal_position):
        self.bus.write4(self.id, ADDR_GOAL_POSITION, int(goal_position))

    def set_velocity_profile(self, value):
        self.bus.write4(self.id, ADDR_PROFILE_VELOCITY, value)

    def set_acceleration_profile(self, value):
        self.bus.write4(self.id, ADDR_PROFILE_ACCELERATION, value)

    def set_goal_pwm(self, value):
        """Caps the motor output in position modes (0..PWM_MAX)."""
        self.bus.write2(self.id, ADDR_GOAL_PWM, int(value))

    def pwm_limit(self):
        return self.bus.read2(self.id, ADDR_PWM_LIMIT)

    def present_load(self):
        """Percent of max torque, signed."""
        return self.bus.read2(self.id, ADDR_PRESENT_LOAD) / 10.0

    def present_voltage(self):
        return self.bus.read2(self.id, ADDR_PRESENT_VOLTAGE) / 10.0

    def present_temperature(self):
        return self.bus.read1(self.id, ADDR_PRESENT_TEMPERATURE)

    def is_moving(self):
        return self.bus.read1(self.id, ADDR_MOVING)

    # 4096 ticks per turn: 1024 = 90 degrees
    def to_angle(self, p):
        return p * 360.0 / TICKS_PER_TURN

    def to_position(self, a):
        return int(round(a * TICKS_PER_TURN / 360.0))
