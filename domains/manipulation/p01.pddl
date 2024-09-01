(define (problem manipulation-01)
(:domain manipulation)
(:objects
living-room kitchen bedroom garage - location
guitar microphone - object)
(:init
(robot-at garage)
(human-at bedroom)
(at guitar bedroom)
(at microphone garage)
(hand-empty)
)
(:goal
(and
(at guitar living-room)
(at microphone living-room)
)
)
(:constraints (and
    (not (and (robot-at bedroom) (human-at bedroom)))
    (not (and (robot-at living-room) (human-at living-room)))
))
)
