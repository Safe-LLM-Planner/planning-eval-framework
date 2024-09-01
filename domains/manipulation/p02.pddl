(define (problem manipulation-02)
(:domain manipulation)
(:objects
living-room kitchen bedroom garage bathroom - location
child-proof-container scissors - object)
(:init
(robot-at kitchen)
(at child-proof-container garage)
(at scissors living-room)
(hand-empty)
)
(:goal
(and
(at scissors bathroom)
)
)
(:constraints (and
    (imply (at scissors bathroom) (at child-proof-container bathroom))
))
)
