# src/robot/robot_interface.py
# Author: Reforge Robotics (Nosa Edoimioya)
# Description: Specific code to create calibration interface for any Python Robot.
# Version: 2.0

# {~.~} START: Axol SDK imports.
import asyncio
import threading
from collections.abc import Mapping
from importlib.resources import as_file, files
from pathlib import Path
from typing import Literal, Optional, Sequence

import numpy as np

# {~.~} Import the robot SDK and required modules here.
from almond_axol.constants import ARM_JOINTS, CAN_LEFT, CAN_RIGHT, urdf_arm_joint_names
from almond_axol.robot import Axol
# {~.~} END: Axol SDK imports.

from reforge_core.hw_interfaces.arm_client import ArmClient
from reforge_core.hw_interfaces.imu_recorder import ImuRecorder

# ------NOTES-----
# 1. Where you see the #{~.~} symbol, you need to make a change. Use Ctrl+F to find all instances.
# The general flow will be the following:
#   a. Import the robot's Python SDK
#   b. Change the BOT_ID, URDF_PATH, ROBOT_MAX_FREQ, and
#      FULL_STRETCH_SHOULDER_ANGLE, FULL_STRETCH_XYZ, FULL_STRETCH_QUAT, and FULL_STRETCH_JOINTS constants
#   c. Change the IS_DEGREES constant if the robot uses degrees instead of radians
#   d. Change the code in the REQUIRED METHODS section to use the robot's SDK
# 2. The REQUIRED METHODS section contains methods that must be implemented for the robot to work with the
#    system identification and calibration workflow. The rest of the methods are pre-defined and should not
#    need to be changed.
# 3. Robot-specific integration points are marked in the REQUIRED METHODS section. {~.~}
# 4. If you opt to use ROS for publishing joint positions, you can use the ros_manager.py file
# in the robots folder. See detailed instructions in that file.

# User constants - EDITS REQUIRED

# {~.~} START: Axol arm configuration.
# ========== BIMANUAL SPECIFIC ==============
# Changing this flag selects the complete arm profile below.
USE_LEFT = True
AXOL_SIDE = "left" if USE_LEFT else "right"
AXOL_CAN_CHANNEL = CAN_LEFT if USE_LEFT else CAN_RIGHT
AXOL_SDK_ARM_ATTRIBUTE = AXOL_SIDE
AXOL_TCP_LINK = f"{AXOL_SIDE}_gripper"
AXOL_JOINT_NAMES = tuple(joint.value for joint in ARM_JOINTS)
AXOL_URDF_JOINT_NAMES = tuple(urdf_arm_joint_names(is_left=USE_LEFT))
BOT_ID = ""
URDF_PATH = f"urdf/axol-{AXOL_SIDE}.urdf"
FULL_STRETCH_XYZ = [0.781526 if USE_LEFT else -0.781526, 0.0, 0.0]
FULL_STRETCH_QUAT = [0.0, -0.7071067812 if USE_LEFT else 0.7071067812, 0.0, 0.7071067812]
FULL_STRETCH_JOINTS = [0.0, -np.pi / 2 if USE_LEFT else np.pi / 2, 0.0, 0.0, 0.0, 0.0, 0.0]
DEFAULT_TCP_PAYLOAD = 0.0
# {~.~} END: Axol arm configuration.

# ========== COMMON PARAMETERS ==============
ROBOT_MAX_FREQ = 240  # {~.~} [CHANGE TO ROBOT'S MAX SAMPLING FREQUENCY] in [Hz]
FULL_STRETCH_POSE_OVERRIDE = None  # {~.~} list of home pose (xyz and quaternion) to override additional height not in base height

# General constants
IS_DEGREES = False  # {~.~} [CHANGE TO TRUE IF ROBOT USES DEGREES]
DATA_LOCATION_PREFIX = "src/robot/data"  # {~.~} [CHANGE TO LOCATION DESIRED - will be robot/DATA_LOCATION_PREFIX/*]
SIM_DATA_LOCATION_PREFIX = str(Path(__file__).resolve().parent / "data" / "sim")

MAX_ROBOT_JOINTS_BANDWIDTH = (
    5.0  # {~.~} Servo motor bandwidth. Leave as is if you don't know [Hz]
)

# {~.~} IMU information
USE_REFORGE_IMU = True
DEFAULT_IMU_COMM_MODE: Literal["ble", "usb", "virtual"] = "usb"
DEFAULT_IMU_RECORD_MODE: Literal["streaming", "logging"] = "streaming"
DEFAULT_IMU_RECORD_FREQUENCY_HZ = ROBOT_MAX_FREQ


class RobotInterface(ArmClient):
    """Provide a concrete robot implementation for system identification and calibration.

    Args:
        robot_ip: Live robot internet protocol address.
        tcp_payload: Optional payload of the robot for NN prediction of
            payload changes.
        tcp_payload_com: Optional 3x1 center of mass location of the
            payload, defined relative to the origin of the TCP [meters].
        local_ip: Local internet protocol address for networked setups.
        sdk_token: Authentication token for the robot software development kit.
        robot_id: Identifier for the robot in the control stack.

    Side Effects:
        Loads the robot model from the configured Unified Robot Description Format file.
        Connects to the robot hardware.

    Raises:
        ValueError: If the simulator sentinel is passed to the hardware adapter.
        RuntimeError: If the robot connection fails or required telemetry is missing.
        ValueError: If reported joint counts do not match the loaded model.

    Preconditions:
        The robot software development kit is installed and the Unified Robot
        Description Format file path is valid.
    """

    def __init__(
        self,
        robot_ip: str,
        local_ip: str = "",
        sdk_token: str = "",
        api_token: str = "",
        robot_id: str = BOT_ID,
        use_reforge_imu: bool = USE_REFORGE_IMU,
        imu_record_mode: Literal["streaming", "logging"] = DEFAULT_IMU_RECORD_MODE,
        imu_comm_mode: Literal["ble", "usb", "virtual"] = DEFAULT_IMU_COMM_MODE,
        imu_record_frequency_hz: float | int = DEFAULT_IMU_RECORD_FREQUENCY_HZ,
        imu_recorder: ImuRecorder | None = None,
        tcp_payload: float = DEFAULT_TCP_PAYLOAD,
        tcp_payload_com: Sequence[float] | None = None,
    ) -> None:
        """Initialize the robot interface and load the URDF model.

        Args:
            robot_ip: Live robot IP address.
            local_ip: Local machine IP address if required by the SDK.
            sdk_token: SDK authentication token.
            api_token: Reforge API token.
            robot_id: Reforge robot ID (most cases) or SDK identifier used by the control stack.
            use_reforge_imu: Whether to use the built-in Reforge IMU backend
                when `imu_recorder` is not supplied.
            imu_record_mode: Reforge IMU acquisition backend used when
                `imu_recorder` is not supplied.
            imu_comm_mode: Reforge IMU communication backend used when
                `imu_recorder` is not supplied.
            imu_record_frequency_hz: Reforge IMU recording frequency [Hz] used
                when `imu_recorder` is not supplied.
            imu_recorder: Optional vendor-specific recorder supplied directly
                by an application or integration test.
            tcp_payload: Payload mass attached at the TCP [kg].
            tcp_payload_com: Optional payload center of mass in TCP coordinates [m].

        Side Effects:
            Loads the URDF model and connects to robot hardware.

        Raises:
            ValueError: If the simulator sentinel is passed to the hardware adapter.
            RuntimeError: If the robot connection fails.
            ValueError: If reported joint counts do not match the URDF.

        Preconditions:
            The URDF file is available and the SDK is installed.
        """
        if robot_ip == "sim":
            raise ValueError(
                "RobotInterface is hardware-only; construct simulator mode "
                "through reforge_core.calibration.run_helpers."
            )

        super().__init__(
            name="My Robot", recording_data_frequency_hz=ROBOT_MAX_FREQ
        )  # {~.~} [Edit with your robot's name and sampling frequency]

        self.max_sampling_frequency_hz = ROBOT_MAX_FREQ
        self.data_folder_prefix = DATA_LOCATION_PREFIX
        self.servo_bandwidth_hz = MAX_ROBOT_JOINTS_BANDWIDTH
        self.calibration_start_joints = FULL_STRETCH_JOINTS
        self.calibration_start_quat = FULL_STRETCH_QUAT
        self.calibration_start_xyz = FULL_STRETCH_XYZ
        self.full_stretch_pose_override = FULL_STRETCH_POSE_OVERRIDE

        # Initialize URDF location
        self.module_dir = files("robot")
        resource = self.module_dir.joinpath(URDF_PATH)
        with as_file(resource) as p:
            self._urdf_path = str(p)
        print(f"URDF Path: {self._urdf_path}")

        # Load robot model from URDF
        if not self.model_is_loaded:
            self.model = self.initialize_model_from_urdf(
                urdf_path=self.urdf_path,
                tcp_payload=tcp_payload,
                tcp_payload_com=tcp_payload_com,
            )
            # Use the model joint count as the ground truth for downstream
            # dynamics calls (the hardware may report extra fixed joints/grippers).
            self.num_joints = self.model.num_joints

        # {~.~} START: Axol state, startup, telemetry, validation, and cleanup.
        self.use_reforge_imu = use_reforge_imu
        self.robot: Axol | None = None
        self._axol_loop: asyncio.AbstractEventLoop | None = None
        self._axol_thread: threading.Thread | None = None
        # {~.~} START: Phase 4 motion state.
        self._axol_motion_enabled = False
        # {~.~} END: Phase 4 motion state.

        # Reforge API and robot ID token is needed for "joint_tracker" product
        # Add it in the CLI with `--identify`
        self.reforge_api_token = api_token
        try:
            # {~.~} Instantiate live robot mode.
            self.robot = Axol(
                left_channel=AXOL_CAN_CHANNEL if USE_LEFT else None,
                right_channel=None if USE_LEFT else AXOL_CAN_CHANNEL,
                left_joints=ARM_JOINTS if USE_LEFT else None,
                right_joints=None if USE_LEFT else ARM_JOINTS,
                loop_hz=ROBOT_MAX_FREQ,
            )
            self._axol_loop = asyncio.new_event_loop()
            self._axol_thread = threading.Thread(
                target=self._axol_loop.run_forever,
                name="axol-event-loop",
                daemon=True,
            )
            self._axol_thread.start()

            # {~.~} Enable ROS control, if necessary.
            # This Axol integration does not require a ROS-control transition.
            self._run_axol(self.robot.connect())
            self._run_axol(
                self.robot.start_telemetry(ROBOT_MAX_FREQ, torque=True)
            )
            self._run_axol(self.robot.wait_for_telemetry())
            active_arm = self.robot.left if USE_LEFT else self.robot.right
            inactive_arm = self.robot.right if USE_LEFT else self.robot.left
            if active_arm is None or inactive_arm is not None:
                raise RuntimeError("Axol did not isolate the selected arm.")

            # {~.~} Unbrake the robot if not operational.
            # Initialization remains non-actuating; torque control is handled later.
            self.id = robot_id
            num_joints_sdk = len(self._get_joint_positions())
            if num_joints_sdk != self.num_joints:
                raise RuntimeError(
                    f"Number of robot joints in URDF ({self.num_joints}) is not "
                    f"equivalent to the number returned by Axol ({num_joints_sdk})."
                )
            self.pose_length = 7

        except BaseException as error:
            self._cleanup_axol_after_error(error)
            if isinstance(error, Exception):
                raise RuntimeError(
                    f"Error getting {robot_ip} operational: {error}"
                ) from error
            raise

        try:
            if not self.imu_manager_is_loaded:
                selected_imu_recorder = imu_recorder
                if selected_imu_recorder is None and not self.use_reforge_imu:
                    selected_imu_recorder = self.create_robot_imu_recorder()
                self.use_reforge_imu = selected_imu_recorder is None
                self.arm_imu_manager = self.initialize_arm_imu_manager(
                    arm_sample_time_s=1.0 / ROBOT_MAX_FREQ,
                    imu_record_mode=imu_record_mode,
                    imu_comm_mode=imu_comm_mode,
                    imu_record_frequency_hz=imu_record_frequency_hz,
                    imu_recorder=selected_imu_recorder,
                )
        except BaseException as error:
            self._cleanup_axol_after_error(error)
            raise

        # {~.~} END: Axol state, startup, telemetry, validation, and cleanup.
    # {~.~} START: Asynchronous Axol helpers and close.
    def _cleanup_axol_after_error(self, error: BaseException) -> None:
        try:
            self.close()
        except BaseException as cleanup_error:
            error.add_note(f"Axol cleanup failed: {cleanup_error}")

    def _run_axol(self, coroutine) -> object:
        if self._axol_loop is None:
            raise RuntimeError("Axol event loop is not initialized.")
        # {~.~} Submit one synchronous SDK call to the persistent Axol loop.
        return asyncio.run_coroutine_threadsafe(coroutine, self._axol_loop).result()

    def _get_joint_positions(self) -> list[float]:
        if self.robot is None:
            raise RuntimeError("Axol robot is not initialized.")
        arm = self.robot.left if USE_LEFT else self.robot.right
        if arm is None:
            raise RuntimeError(f"No {AXOL_SIDE} Axol arm is available.")
        return np.asarray(arm.positions, dtype=float)[: self.num_joints].tolist()

    def close(self) -> None:
        """Stop recording and cleanly close the Axol async loop."""
        # {~.~} START: Simplified Phase 4 close.
        if self.robot is None:
            return
        self.stop_recording()
        # {~.~} Axol disables the selected seven-joint arm and closes its buses.
        self._run_axol(self.robot.disable())
        self._axol_motion_enabled = False  # {~.~} Clear ownership after success.
        if self._axol_loop is not None:
            self._axol_loop.call_soon_threadsafe(self._axol_loop.stop)
        if self._axol_thread is not None:
            self._axol_thread.join()
            if self._axol_thread.is_alive():
                raise RuntimeError("Axol event-loop thread did not stop.")
        if self._axol_loop is not None:
            self._axol_loop.close()
        self.robot = None
        self._axol_loop = None
        self._axol_thread = None
        # {~.~} END: Simplified Phase 4 close.

    # {~.~} END: Asynchronous Axol helpers and close.

    def create_robot_imu_recorder(self) -> ImuRecorder:
        """Create the robot-native IMU adapter used when Reforge IMU is disabled.

        Robot integrations should override this method and return an
        `ImuRecorder` that converts SDK samples into `IMUState` values in SI
        units. The recorder must emit Unix epoch timestamps aligned with arm
        state timestamps, either because both originate from one controller
        clock or because `prepare()` estimates and applies their offset.

        Returns:
            `ImuRecorder` backed by the robot vendor's native IMU API.

        Raises:
            NotImplementedError: If this robot template has not implemented a
                native IMU adapter.
        """
        raise NotImplementedError(
            "use_reforge_imu=False requires RobotInterface."
            "create_robot_imu_recorder() to return a vendor-specific "
            "ImuRecorder."
        )

    @property
    def in_sim_mode(self) -> bool:
        """Return whether the interface is running in simulator mode.

        Returns:
            `bool` always false for the hardware adapter.
        """
        return False

    @property
    def urdf_path(self) -> str:
        """Return the absolute path to the URDF file.

        Returns:
            `str` path to the URDF file.
        """
        return self._urdf_path

    # {~.~} REQUIRED METHODS
    def command_move_j(
        self,
        target_joints: np.ndarray | list[float] | tuple[float, ...],
        *,
        speed: float = 50.0,
        wait: bool = True,
    ) -> int:
        """Send a blocking/non-blocking point-to-point joint command using the
        robot's native position control interface.

        Args:
            target_joints: Target joint positions [rad] as a list or array.
            speed: Speed percentage for the motion, if supported by the robot. Default is 50%.
            wait: If `True`, block until the motion is complete. If `False`, return immediately after sending the command.

        Returns:
            An integer status code from the robot's command interface, if applicable.
            If the robot does not provide a status code, return 0 for success or raise an exception for failure.
        """
        if IS_DEGREES:
            target_joints = list(np.rad2deg(angle) for angle in target_joints)

        arm = self._require_connected_arm()  # noqa: F841
        # {~.~} Send a joint target through the selected Axol arm.
        # {~.~} Replace the placeholder return after implementation and testing.
        return 1

    def command_move_pose(
        self,
        target_quat: np.ndarray | list[float],
        target_xyz: np.ndarray | list[float],
        *,
        speed: float = 50.0,
        wait: bool = True,
        locked_joints: Mapping[int, float] | None = None,
    ) -> int:
        """Send a blocking/non-blocking point-to-point pose command using the
        robot's native position control interface.

        Args:
            target_quat: Target TCP orientation as a quaternion `[qx, qy, qz, qw]` [-] in the robot's base frame.
            target_xyz: Target TCP position `[x, y, z]` [m] in the robot's base frame.
            speed: Speed percentage for the motion, if supported by the robot. Default is 50%.
            wait: If `True`, block until the motion is complete. If `False`, return immediately after sending the command.
            locked_joints: Simulator-only joint-index to fixed position map [rad].

        Returns:
            An integer status code from the robot's command interface, if applicable.
            If the robot does not provide a status code, return 0 for success or raise an exception for failure.
        """
        if locked_joints is not None:
            raise RuntimeError("locked_joints is only supported in simulator mode.")

        arm = self._require_connected_arm()  # noqa: F841
        # {~.~} Send a Cartesian target through the selected Axol arm.
        # {~.~} Replace the placeholder return after implementation and testing.
        return 1

    # {~.~} START: Shared Axol joint-target validation.
    @staticmethod
    def _validate_joint_target(
        target_joints: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        try:
            target = np.asarray(target_joints, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Axol joint targets must be numeric.") from exc
        if target.shape != (len(ARM_JOINTS),):
            raise ValueError(f"Expected {len(ARM_JOINTS)} joint targets.")
        if not np.all(np.isfinite(target)):
            raise ValueError("Axol joint targets must be finite.")
        return target

    # {~.~} END: Shared Axol joint-target validation.
    # {~.~} START: Phase 5 servo command method.
    def command_servo_j(
        self,
        target_joints: np.ndarray | list[float],
        *,
        wait: bool = False,
    ) -> int:
        """Send one servo command in radians.

        This function will be used to stream a sequence of positions in a for loop in the calibration routine.

        Args:
            target_joints: Target joint position [rad] as a list or array.
            wait: If `True`, block until the motion is complete. If `False`, return immediately after sending the command.

        Returns:
            An integer status code from the robot's command interface, if applicable.
            If the robot does not provide a status code, return 0 for success or raise an exception for failure.
        """
        # {~.~} START: Phase 5 Axol servo command.
        robot = self._require_connected_arm()
        del wait  # Axol has no target-settled acknowledgement.
        target = self._validate_joint_target(target_joints)
        if robot.fault is not None:
            raise RuntimeError(f"Axol realtime core faulted: {robot.fault}")
        if robot.limp is not None:
            raise RuntimeError(f"Axol realtime core is limp: {robot.limp}")
        command = np.zeros(8, dtype=np.float32)
        command[: len(ARM_JOINTS)] = target
        self._run_axol(
            robot.motion_control(**{AXOL_SDK_ARM_ATTRIBUTE: command})
        )
        # {~.~} END: Phase 5 Axol servo command.
        return 0  # {~.~} Phase 5: command submitted successfully.

    # {~.~} END: Phase 5 servo command method.
    def enter_position_mode(self) -> Optional[int | None]:
        """
        Ensure the controller is in point-to-point position mode before issuing queued P2P moves.

        Returns:
            the mode/state codes so they can be inspected when debugging.
        """
        # {~.~} START: Simplified Phase 4 position entry.
        arm = self._require_connected_arm()  # noqa: F841
        if not self._axol_motion_enabled:
            # {~.~} Axol uses one realtime impedance controller for both modes.
            self._run_axol(self.robot.enable())
            self._axol_motion_enabled = True  # {~.~} Set ownership after success.
        if self.robot.fault is not None:
            raise RuntimeError(f"Axol realtime core faulted: {self.robot.fault}")  # {~.~}
        if self.robot.limp is not None:
            raise RuntimeError(f"Axol realtime core is limp: {self.robot.limp}")  # {~.~}
        # {~.~} END: Simplified Phase 4 position entry.
        return 0  # {~.~} Axol has no distinct position/servo mode code.

    def enter_servo_mode(self) -> Optional[int | None]:
        """Ensure the controller is set to servo control mode.

        Returns:
            the mode/state codes so they can be inspected when debugging.
        """
        return self.enter_position_mode()  # {~.~} Axol shares one realtime control mode.

    def supports_teaching_mode(self) -> bool:
        """Return whether the robot supports manual teaching mode.

        Override this method when the robot SDK supports hand-guided teaching.

        Returns:
            `bool` indicating whether manual teaching mode is implemented.
        """
        return False  # {~.~} Phase 3: teaching command is deferred to a later phase.

    def enter_teaching_mode(self) -> Optional[int | None]:
        """Ensure the controller is set to manual teaching mode.

        Override this method with the robot SDK's teaching-mode command.

        Returns:
            Vendor-specific mode/state code when available.
        """
        arm = self._require_connected_arm()  # noqa: F841

        # {~.~} Enable manual teaching mode using the Axol SDK.

        # {~.~} Return 0 for success - edit after implementation and testing
        return 1

    def supports_flange_button(self) -> bool:
        """Return whether the robot exposes a readable flange button.

        Override this method when the robot SDK exposes a button or equivalent
        operator input near the tool flange.

        Returns:
            `bool` indicating whether flange-button reads are implemented.
        """
        return False  # {~.~} Phase 3: Axol exposes no documented flange-button input.

    def read_flange_button_pressed(self) -> bool:
        """Return whether the flange button is currently pressed.

        Override this method with the robot SDK's flange-button read.

        Returns:
            `bool` indicating the current flange-button state.
        """
        raise NotImplementedError("Axol does not expose a flange-button input.")  # {~.~} Phase 3: unsupported.

    def get_joint_state(self) -> tuple[list[float], list[float], list[float]]:
        """Return one joint state sample as ``(q, qd, tau)``.

        Returns:
            Tuple of three lists: joint positions `q` [rad], velocities `qd` [rad/s],
            and efforts/currents `tau` [SDK units].
        """
        arm = self._require_connected_arm()  # noqa: F841
        q: list[float] = []
        qd: list[float] = []
        tau: list[float] = []

        # {~.~} Read joint position, velocity, and effort from the selected Axol arm.

        if not IS_DEGREES:
            q = [np.deg2rad(value) for value in q]
            qd = [np.deg2rad(value) for value in qd]

        return q, qd, tau

    def get_tcp_pose(self) -> list[float]:
        """Return TCP pose as ``[x, y, z, qx, qy, qz, qw]``.

        Returns:
            List of 7 floats representing the TCP pose in meters for positions
            and unitless normalized for quaternions.
        """
        position: list[float] = []
        quat: list[float] = []
        arm = self._require_connected_arm()  # noqa: F841

        # {~.~} Read and normalize the selected Axol TCP pose.

        # Return tooltip pose as a list
        return [*position, *quat]

    # {~.~} END OF REQUIRED METHODS

    # {~.~} OPTIONAL OVERRIDES
    def command_joint_trajectory(
        self,
        time_data: Sequence[float],
        position_stream: Sequence[Sequence[float]],
        velocity_stream: Sequence[Sequence[float]] | None = None,
        acceleration_stream: Sequence[Sequence[float]] | None = None,
        Ts: float = 1.0 / ROBOT_MAX_FREQ,
    ) -> list[tuple[float, list[float]]]:
        """Send one complete joint trajectory and return command timestamps.

        *OVERRIDE* this method when the robot SDK supports a native trajectory
        upload/stream API, requires velocity or acceleration feedforward, or
        needs controller-specific readiness checks before publishing a full
        trajectory. The default implementation delegates to `ArmClient`, which
        streams each sample through `command_servo_j()` at the requested timing.

        Args:
            time_data: Command timestamps [s].
            position_stream: Joint position commands [rad].
            velocity_stream: Optional joint velocity commands [rad/s].
            acceleration_stream: Optional joint acceleration commands [rad/s^2].
            Ts: Sampling time [s].

        Returns:
            `list[tuple[float, list[float]]]` host publish timestamps [s] and
            joint position commands [rad].
        """
        # {~.~} OPTIONAL: Only override this method if the robot SDK has a
        # native trajectory upload/stream API or requires special handling for
        # velocity/acceleration feedforward. Otherwise, the default
        # implementation in `ArmClient` will stream each sample using
        # `command_servo_j()` at the specified timing.
        return super().command_joint_trajectory(
            time_data=time_data,
            position_stream=position_stream,
            velocity_stream=velocity_stream,
            acceleration_stream=acceleration_stream,
            Ts=Ts,
        )

    # {~.~} END OF OPTIONAL OVERRIDES
