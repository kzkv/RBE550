# Tom Kazakov
# RBE 550
# Assignment 0
# Turtles: implement Victor Sierra search pattern

# This is a dirty try/catch, but economical implementation to handle exceptions in `turtle` and `tkinter`
# getting raised because of the interrupted loop in the `search` function.

import turtle

leg = 100  # Radius of the search pattern
turn_angle = 120
next_search_angle = 30
turtle.mode("logo")  # Use 0 as the north direction
t = turtle.Turtle()
screen = turtle.Screen()
screen.setup(width=500, height=500)


def search():
    try:
        for _ in range(3):
            # Perform the first two fixed legs of the search
            t.forward(leg)
            t.right(turn_angle)
            t.forward(leg)

            # Technically, after the first two fixed legs of the search, we need to return to the datum.
            # Considering there is no drift in this implementation (ideal conditions): moving in equilateral triangle.
            t.right(turn_angle)
            t.forward(leg)
        t.right(next_search_angle)
    except turtle.Terminator:  # Sufficient catch-all to cover the interrupted loop exceptions.
        pass
    screen.ontimer(search, 1000)  # Invoke another with a delay for me to take a screenshot


try:
    search()
    screen.exitonclick()
except Exception:  # ew
    pass
