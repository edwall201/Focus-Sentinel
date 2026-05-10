from luma.core.interface.serial import i2c
from luma.core.render import canvas
from luma.emulator.device import pygame
import time

device = pygame(width=128, height=64)
def draw_ui(focus_time, distraction_time, status = "FOCUSING"):
    with canvas(device) as draw:
        draw.rectangle(device.bounding_box, outline="white", fill="black")
        draw.text((10, 5), f"STATUS: {status}", fill="white")
        draw.line((0, 20, 128, 20), fill="white")
        draw.text((10, 30), f"Focus: {focus_time}s", fill="white")
        draw.text((10, 45), f"Distractions: {distraction_time}s", fill="white")
try:
    start_time = time.time()
    count = 0

    while True:
        elapsed = int(time.time() - start_time)
        draw_ui(elapsed, count, status="WORKING" if elapsed % 10 < 7 else "PHONE!")
        
        if elapsed % 10 == 7:
            count += 1
            
        time.sleep(1)
except KeyboardInterrupt:
    pass