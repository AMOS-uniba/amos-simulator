MAKEFLAGS += --no-builtin-rules --no-builtin-variables

# The chain, per meteor and observer:
#
#   simulate.py   a meteoroid's flight            -> output/<meteor>/meteor.yaml
#   observe.py    what one station would see       -> output/<meteor>/<observer>.yaml
#   render.py     frames of that, with the sky      -> output/<meteor>/<observer>/NNN.png
#   ffmpeg        a video Kvant can reanalyse       -> output/<meteor>/<observer>/video.mp4
#
# so `make output/18/ago/video.mp4` builds the lot. Run through uv, like the rest of the
# ecosystem: `uv sync` first.
#
# The projection is what makes this worth doing. Render through a plate the Borovicka model can
# express and a fit will recover it exactly, leaving nothing for a kernel correction to find; the
# interesting experiment renders through something it cannot.

UV := uv run

.PHONY: all clean
all: ;

# Nothing here is a by-product. Make deletes the intermediates of a chain it built in one go, and
# output/<meteor>/<observer>.yaml is the ground truth every benchmark is scored against -- the one
# file in the whole pipeline that cannot be recovered from anything downstream of it.
.SECONDARY:

output/%/video.mp4: output/%/000.png
	cd $(dir $@) && ffmpeg -y -framerate 20 -pattern_type glob -i '*.png' \
		-c:v libx264 -qp 16 -pix_fmt gray video.mp4

# <meteor>/<observer>
output/%/000.png: render.py default.yaml config/renderers/default.yaml output/%.yaml
	$(eval words := $(subst /, ,$*))
	$(eval meteor := $(word 1,$(words)))
	$(eval observer := $(word 2,$(words)))
	mkdir -p $(dir $@)
	$(UV) ./render.py default.yaml output/$(meteor)/$(observer).yaml \
		config/renderers/default.yaml output/$(meteor)/$(observer)/ \
		-p config/projections/zero.yaml -j 20

# The observation depends on the flight, which is one directory up from it -- a prerequisite that
# has to be computed from the stem, and a pattern rule only expands its prerequisites once. Hence
# second expansion and the doubled dollars.
.SECONDEXPANSION:

# <meteor>/<observer>: what that station saw, in alt-az -- the ground truth a fit is scored against
output/%.yaml: observe.py default.yaml output/$$(word 1,$$(subst /, ,$$*))/meteor.yaml
	$(eval words := $(subst /, ,$*))
	$(eval meteor := $(word 1,$(words)))
	$(eval observer := $(word 2,$(words)))
	mkdir -p $(dir $@)
	$(UV) ./observe.py default.yaml output/$(meteor)/meteor.yaml \
		config/observers/$(observer).yaml -o $@

# <meteor>: the flight itself. Matched in preference to the rule above for output/X/meteor.yaml,
# because make takes the pattern that leaves the shorter stem -- 'X' rather than 'X/meteor'.
output/%/meteor.yaml: simulate.py default.yaml config/meteors/%.yaml
	mkdir -p $(dir $@)
	$(UV) ./simulate.py default.yaml config/meteors/$*.yaml -o $@

clean:
	rm -rf output/*/
