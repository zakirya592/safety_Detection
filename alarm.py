import pygame
import os

class Alarm:

    def __init__(self, sound_file="alarm.wav"):

        pygame.mixer.init()

        if not os.path.isfile(sound_file):
            raise FileNotFoundError(f"{sound_file} not found")

        self.sound_file = sound_file

    def play(self):

        if not pygame.mixer.music.get_busy():

            pygame.mixer.music.load(self.sound_file)
            pygame.mixer.music.play(-1)   # Loop continuously

    def stop(self):

        pygame.mixer.music.stop()