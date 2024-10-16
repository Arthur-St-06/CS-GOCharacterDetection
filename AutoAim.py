from shutil import move
from time import sleep
import pydirectinput

pydirectinput.moveTo(0, 0)

import pyautogui
print(pyautogui.position())  # where it moved before importing
pydirectinput.moveTo(0, 0)
print(pyautogui.position()) 

while True:   
    sleep(5)   
    #x, y = pyautogui.locateCenterOnScreen("ball.jpg", confidence = 0.8)    
    print(pyautogui.position()) 
    pydirectinput.moveTo(0, 0)
    print(pyautogui.position()) 