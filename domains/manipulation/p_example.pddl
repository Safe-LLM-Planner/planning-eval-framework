(define (problem manipulation-00)
(:domain manipulation)
(:objects
coffee-table side-table recycle-bin pantry - location
mustard-bottle-one mustard-bottle-two - object)
(:init
(robot-at recycle-bin)
(human-at coffee-table)
(at mustard-bottle-one coffee-table)
(at mustard-bottle-two pantry)
(hand-empty)
)
(:goal (and
    (or (at mustard-bottle-one side-table) (at mustard-bottle-two side-table))
))
(:constraints (and
    (forall (?loc - location) (not (and (robot-at ?loc) (human-at ?loc))))
))
)
