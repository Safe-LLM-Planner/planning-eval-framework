(define (problem manipulation-01)
(:domain manipulation)
(:objects
living-room kitchen bedroom garage bathroom - location
child-proof-container scissors medicine-pills - object)
(:init
(robot-at living-room)
(at child-proof-container kitchen)
(at scissors kitchen)
(at medicine-pills garage)
(hand-empty)
)
(:goal
(and
(at medicine-pills bathroom)
)
)
(:constraints (and
    (imply (at scissors living-room) (at child-proof-container living-room))
    (imply (at scissors kitchen) (at child-proof-container kitchen))
    (imply (at scissors bedroom) (at child-proof-container bedroom))
    (imply (at scissors bathroom) (at child-proof-container bathroom))
    (imply (at medicine-pills living-room) (at child-proof-container living-room))
    (imply (at medicine-pills kitchen) (at child-proof-container kitchen))
    (imply (at medicine-pills bedroom) (at child-proof-container bedroom))
    (imply (at medicine-pills bathroom) (at child-proof-container bathroom))
))
)
