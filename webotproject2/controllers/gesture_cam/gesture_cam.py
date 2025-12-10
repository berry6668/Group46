# E:\webotproject2\controllers\gesture_cam\gesture_cam.py
from controller import Robot, Keyboard
import socket
import select
import time
import csv
import os

HOST = '0.0.0.0'
PORT = 10020

robot = Robot()
time_step = int(robot.getBasicTimeStep())

# ======== Keyboard Initialization ========
keyboard = robot.getKeyboard()
keyboard.enable(time_step)

# ======== Motor Initialization ========
left_motor = robot.getMotor('left wheel motor')
right_motor = robot.getMotor('right wheel motor')

left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# ======== Wheel Position Sensors (for real speed estimation) ========
left_ps = robot.getPositionSensor('left wheel sensor')
right_ps = robot.getPositionSensor('right wheel sensor')
left_ps.enable(time_step)
right_ps.enable(time_step)

# e-puck wheel radius (approx. 20.5mm)
WHEEL_RADIUS = 0.0205  # m

prev_left_pos = left_ps.getValue()
prev_right_pos = right_ps.getValue()

# "Stuck detection" parameters
stuck_counter = 0
STUCK_STEPS_THRESHOLD = 5        # Number of consecutive steps with almost no movement → collision
COMMAND_SPEED_THRESHOLD = 0.1    # Considered as "moving" when above this command speed (rad/s)
ACTUAL_SPEED_THRESHOLD = 0.1     # Actual linear speed below this → considered almost stopped (m/s)

# ======== LED Initialization (e-puck built-in led0 ~ led9) ========
leds = {}  # Use dict to avoid index issues

for i in range(10):   # led0 ~ led9
    name = f"led{i}"
    try:
        dev = robot.getLED(name)
        leds[i] = dev
    except Exception:
        print(f"[WARN] No device named {name}")

def set_all_leds(value: int):
    """Set all detected LEDs to 0 or 1"""
    for led in leds.values():
        led.set(value)

# LED feedback enable: True = enabled, False = disabled
WITH_LED = True

def update_led_by_command(cmd: str):
    """
    Illuminate different LED patterns according to current command:
    STOP        -> Body LED (assumed led8)
    FORWARD     -> All ring LEDs
    TURN_LEFT   -> Left half ring
    TURN_RIGHT  -> Right half ring
    BACKWARD    -> Rear LEDs
    SPEED_UP    -> Front LEDs (assumed led9)
    SLOW_DOWN   -> Middle LEDs
    """
    cmd = cmd.strip().upper()

    # If LED feedback is disabled, turn off all LEDs and return
    if not WITH_LED:
        set_all_leds(0)
        return

    set_all_leds(0)  # Turn off all LEDs first

    if cmd == "STOP":
        if 8 in leds:
            leds[8].set(1)

    elif cmd == "FORWARD":
        for i in range(8):   # led0~7 ring
            if i in leds:
                leds[i].set(1)

    elif cmd == "BACKWARD":
        for i in [4, 5, 6, 7]:
            if i in leds:
                leds[i].set(1)

    elif cmd == "TURN_LEFT":
        for i in [5, 6, 7]:
            if i in leds:
                leds[i].set(1)

    elif cmd == "TURN_RIGHT":
        for i in [1, 2, 3]:
            if i in leds:
                leds[i].set(1)

    elif cmd == "SPEED_UP":
        for i in [1, 7, 0]:
            if i in leds:
                leds[i].set(1)

    elif cmd == "SLOW_DOWN":
        for i in [3, 4, 5]:
            if i in leds:
                leds[i].set(1)

# ======== Speed & State Variables ========
base_speed_default = 3.0
turn_speed_default = 2.0

base_speed = base_speed_default
turn_speed = turn_speed_default
motion_state = "STOP"

# ======== Experiment Mode & Timing ========
CONTROL_MODE = "GESTURE"   # Or "KEYBOARD" for keyboard trials

task_running = False
start_time = None
end_time = None

PARTICIPANT_ID = "P01"
TRIAL_ID = 1

# Collision & parking flags
collision_happened = False
parking_success = False

# Result files
RESULT_TIME_FILE = "results_time.csv"
RESULT_TRIAL_FILE = "results_trials.csv"

# ======== TCP Server Initialization (Gesture Client) ========
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
server_socket.listen(1)
server_socket.setblocking(False)

print(f"[Controller] Gesture control server started: {HOST}:{PORT}")

client_conn = None
client_addr = None

def handle_command(cmd: str):
    """Unified command handler for network and keyboard inputs"""
    global base_speed, turn_speed, motion_state
    cmd = cmd.strip().upper()

    if cmd in {"FORWARD", "STOP", "TURN_LEFT", "TURN_RIGHT", "BACKWARD"}:
        motion_state = cmd
        print(f"[Controller] Motion state set to: {motion_state}")
        update_led_by_command(cmd)

    elif cmd == "SPEED_UP":
        base_speed = min(base_speed + 1.0, 6.28)
        turn_speed = min(turn_speed + 0.5, 6.28)
        print(f"[Controller] Speed increased: base={base_speed:.2f}, turn={turn_speed:.2f}")
        update_led_by_command(cmd)

    elif cmd == "SLOW_DOWN":
        base_speed = max(base_speed - 1.0, 0.0)
        turn_speed = max(turn_speed - 0.5, 0.0)
        print(f"[Controller] Speed decreased: base={base_speed:.2f}, turn={turn_speed:.2f}")
        update_led_by_command(cmd)

    elif cmd == "EMERGENCY_STOP":
        motion_state = "STOP"
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)
        print("[Controller] Emergency stop activated!")
        update_led_by_command("STOP")

    else:
        print(f"[Controller] Unknown command: {cmd}")

# ======== Main Loop ========
while robot.step(time_step) != -1:
    # ======== Handle network connection (gesture client) ========
    if client_conn is None:
        try:
            conn, addr = server_socket.accept()
            conn.setblocking(False)
            client_conn = conn
            client_addr = addr
            print(f"[Controller] Client connected: {client_addr}")
            update_led_by_command(motion_state)
        except BlockingIOError:
            pass
    else:
        try:
            ready_to_read, _, _ = select.select([client_conn], [], [], 0)
            if ready_to_read:
                data = client_conn.recv(1024)
                if data:
                    cmd_str = data.decode('utf-8')
                    handle_command(cmd_str)
                else:
                    print("[Controller] Client disconnected")
                    client_conn.close()
                    client_conn = None
                    client_addr = None
                    motion_state = "STOP"
                    left_motor.setVelocity(0.0)
                    right_motor.setVelocity(0.0)
                    update_led_by_command("STOP")
        except (BlockingIOError, InterruptedError):
            pass
        except OSError:
            print("[Controller] Socket error, closing connection")
            try:
                client_conn.close()
            except Exception:
                pass
            client_conn = None
            client_addr = None
            motion_state = "STOP"
            left_motor.setVelocity(0.0)
            right_motor.setVelocity(0.0)
            update_led_by_command("STOP")

    # ======== Keyboard Input Handling (WASD + J/K + B/N + P + L) ========
    key = keyboard.getKey()
    while key != -1:
        if key in (ord('W'), ord('w')):
            handle_command("FORWARD")
        elif key in (ord('A'), ord('a')):
            handle_command("TURN_LEFT")
        elif key in (ord('S'), ord('s')):
            handle_command("BACKWARD")
        elif key in (ord('D'), ord('d')):
            handle_command("TURN_RIGHT")
        elif key in (ord('J'), ord('j')):
            handle_command("SPEED_UP")
        elif key in (ord('K'), ord('k')):
            handle_command("EMERGENCY_STOP")

        elif key in (ord('B'), ord('b')):
            if not task_running:
                task_running = True
                start_time = time.time()
                collision_happened = False
                parking_success = False
                print("[Exp] Task started (PARTICIPANT_ID={}, MODE={}, TRIAL={})"
                      .format(PARTICIPANT_ID, CONTROL_MODE, TRIAL_ID))
            else:
                print("[Exp] Task already running, start ignored")

        elif key in (ord('P'), ord('p')):
            if task_running:
                parking_success = True
                print("[Exp] Parking success marked")
            else:
                print("[Exp] No active task, P key ignored")

        elif key in (ord('L'), ord('l')):
            WITH_LED = not WITH_LED
            if not WITH_LED:
                set_all_leds(0)
                print("[Exp] LED feedback disabled")
            else:
                print("[Exp] LED feedback enabled, refreshing LED state")
                update_led_by_command(motion_state)

        elif key in (ord('N'), ord('n')):
            if task_running:
                end_time = time.time()
                duration = end_time - start_time
                task_running = False
                print(f"[Exp] Task ended, duration {duration:.3f} seconds")

                try:
                    write_header = not os.path.exists(RESULT_TIME_FILE)
                    with open(RESULT_TIME_FILE, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        if write_header:
                            writer.writerow(["participant", "mode", "trial", "duration_sec"])
                        writer.writerow([PARTICIPANT_ID, CONTROL_MODE, TRIAL_ID, duration])
                    print("[Exp] Written to", RESULT_TIME_FILE)
                except Exception as e:
                    print("[Exp] Failed to write", RESULT_TIME_FILE, ":", e)

                try:
                    write_header = not os.path.exists(RESULT_TRIAL_FILE)
                    with open(RESULT_TRIAL_FILE, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        if write_header:
                            writer.writerow([
                                "participant",
                                "mode",
                                "trial",
                                "duration_sec",
                                "collision",
                                "parking"
                            ])
                        writer.writerow([
                            PARTICIPANT_ID,
                            CONTROL_MODE,
                            TRIAL_ID,
                            duration,
                            int(collision_happened),
                            int(parking_success),
                        ])
                    print("[Exp] Written to", RESULT_TRIAL_FILE)
                except Exception as e:
                    print("[Exp] Failed to write", RESULT_TRIAL_FILE, ":", e)

                TRIAL_ID += 1
            else:
                print("[Exp] No active task, N key ignored")

        key = keyboard.getKey()

    # ======== Motor Control According to Motion State ========
    if motion_state == "STOP":
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)
        commanded_speed_mag = 0.0
    elif motion_state == "FORWARD":
        left_motor.setVelocity(base_speed)
        right_motor.setVelocity(base_speed)
        commanded_speed_mag = abs(base_speed)
    elif motion_state == "BACKWARD":
        left_motor.setVelocity(-base_speed)
        right_motor.setVelocity(-base_speed)
        commanded_speed_mag = abs(base_speed)
    elif motion_state == "TURN_LEFT":
        left_motor.setVelocity(-turn_speed)
        right_motor.setVelocity(turn_speed)
        commanded_speed_mag = abs(turn_speed)
    elif motion_state == "TURN_RIGHT":
        left_motor.setVelocity(turn_speed)
        right_motor.setVelocity(-turn_speed)
        commanded_speed_mag = abs(turn_speed)
    else:
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)
        commanded_speed_mag = 0.0

    # ======== Real Speed Estimation & Collision Detection ========
    left_pos = left_ps.getValue()
    right_pos = right_ps.getValue()

    dl = left_pos - prev_left_pos
    dr = right_pos - prev_right_pos

    prev_left_pos = left_pos
    prev_right_pos = right_pos

    dt = time_step / 1000.0
    lin_speed = (dl + dr) * 0.5 * WHEEL_RADIUS / dt

    if task_running and motion_state != "STOP" and commanded_speed_mag > COMMAND_SPEED_THRESHOLD:
        if abs(lin_speed) < ACTUAL_SPEED_THRESHOLD:
            stuck_counter += 1
            if stuck_counter >= STUCK_STEPS_THRESHOLD and not collision_happened:
                collision_happened = True
                print("[Exp] Collision detected: commanded to move but speed dropped")
        else:
            stuck_counter = 0
    else:
        stuck_counter = 0
