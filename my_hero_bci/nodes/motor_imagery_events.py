#!/usr/bin/env python3
import rospy
from rosneuro_msgs.msg import NeuroEvent, NeuroHeader
from std_msgs.msg import Header
import pygame
import time

# Duraciones (segundos)
REST_TIME = 5
OPEN_TIME = 5
CLOSE_TIME = 5
REPETITIONS = 25

def publish_event(pub, event_id, description):
    """Publica un mensaje NeuroEvent en /neuroevent"""
    msg = NeuroEvent()
    msg.header = Header()
    msg.header.stamp = rospy.Time.now()
    msg.neuroheader = NeuroHeader()
    msg.neuroheader.seq = int(time.time()*1000) % 100000  # simple contador temporal
    msg.version = "1.0"
    msg.event = event_id
    msg.duration = 0.0
    msg.family = 0
    msg.description = description
    pub.publish(msg)

def show_text(screen, text, color=(255,255,255)):
    """Muestra texto grande en pantalla"""
    screen.fill((0,0,0))
    font = pygame.font.Font(None, 100)
    rendered = font.render(text, True, color)
    rect = rendered.get_rect(center=(screen.get_width()/2, screen.get_height()/2))
    screen.blit(rendered, rect)
    pygame.display.flip()

def main():
    rospy.init_node('motor_imagery_event_publisher')
    pub = rospy.Publisher('/neuroevent', NeuroEvent, queue_size=10)
    rate = rospy.Rate(10)

    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Motor Imagery - BCI")

    show_text(screen, "Preparado...", (0,255,0))
    rospy.sleep(3)

    for trial in range(REPETITIONS):
        for event_id, label, duration in [
            (0, "REPOSO", REST_TIME),
            (1, "ABRIR MANO", OPEN_TIME),
            (2, "CERRAR MANO", CLOSE_TIME)
        ]:
            show_text(screen, label)
            publish_event(pub, event_id, label)
            start_time = time.time()

            while time.time() - start_time < duration:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                rate.sleep()

    show_text(screen, "FIN DE LA SESIÓN", (0,255,0))
    rospy.sleep(3)
    pygame.quit()

if __name__ == "__main__":
    main()

