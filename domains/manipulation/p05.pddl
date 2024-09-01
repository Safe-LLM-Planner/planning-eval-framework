(define (problem manipulation-01)
(:domain manipulation)
(:objects
living-room kitchen bedroom garage bathroom - location
sterile-pad adhesive-tape scissors - object)
(:init
(robot-at kitchen)
(at sterile-pad garage)
(at adhesive-tape bedroom)
(at scissors living-room)
(hand-empty)
)
(:goal
(and
(at scissors bathroom)
)
)
(:constraints (and
    (or (not (at scissors bathroom)) (and (at sterile-pad bathroom) (at adhesive-tape bathroom)))
))
)
