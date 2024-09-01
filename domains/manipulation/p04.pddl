(define (problem manipulation-01)
(:domain manipulation)
(:objects
living-room kitchen bedroom garage bathroom - location
first-aid-kit scissors - object)
(:init
(robot-at kitchen)
(at first-aid-kit garage)
(at scissors living-room)
(hand-empty)
)
(:goal
(and
(at scissors bathroom)
)
)
(:constraints (and
    (or (not (at scissors bathroom)) (at first-aid-kit bathroom))
))
)
