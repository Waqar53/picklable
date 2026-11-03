# Labelling guide (class `plastic`)

One page, so every reviewer labels the same way. Auto-labels follow the same rules, and
review is where they get enforced.

## Label

- **Every piece of plastic contaminant**: bags, film, cling film, wrappers, chip packets,
  compostable-looking bags that aren't certified, hard plastic fragments.
- **Only the visible part.** Film half-buried under peels: outline what you can see, not where
  you think the rest is. A bag split by an onion skin lying on top is still one instance if it
  is one piece of plastic.
- **Plastic wrapped around food**: label the plastic, not the food inside.
- **Transparent film**: label it even when the background shows through. Follow the
  highlights and the edge where texture changes.
- **Fragments**: label every piece with a long side of at least **1 cm** (about 12 px at 1080p
  from 60 cm). Smaller pieces: don't label. Note the cut-off in the dataset version notes.

## Don't label

- **Look-alikes**: onion skin, wet paper, banana leaves, garlic peel, eggshell membrane,
  aluminium foil (a separate class later, not `plastic`). They are what teaches the model
  precision; a cleared-tray frame full of them is a valuable negative image.
- Reflections and shadows of plastic.

## Review queue (route:review)

- Accept the auto-label if its outline is within about 2 px of the item edge all the way round.
- Fix it rather than redraw it: drag vertices, cut off a neighbour it swallowed.
- If the image has an unlabelled plastic item (tag `incomplete`), add it. A frame with
  missing labels is worse than no frame.

## Hard examples (route:hard)

- `gripper_empty` + `no_change_at_grasp`: the robot went for something and nothing was there
  to pick. Check the grasp point first: if it is a look-alike, the frame is a perfect
  false-positive example. Label the real plastic only.
- `gripper_holding_but_no_visible_change`: usually transparent film. Label it carefully;
  these are the rarest and most valuable labels.
