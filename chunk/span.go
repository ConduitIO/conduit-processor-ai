// Copyright © 2026 Meroxa, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package chunk

import "fmt"

// span is a half-open range [start, end) of rune indices into a document's
// []rune slice. Every span this package produces is a valid,
// order-preserving slice bound (0 <= start <= end <= len(runes)) and every
// strategy's output spans are contiguous and non-overlapping — see doc.go's
// "chunks are always contiguous substrings of the source document" note.
type span struct {
	start, end int
}

func (s span) len() int { return s.end - s.start }

// splitSpans dispatches to the configured strategy and returns the spans
// its chunks cover, in source order. Returns nil for empty input (0 chunks
// — see doc.go's fan-out note) or an error for a strategy name Config.
// Validate should already have rejected (defensive; unreachable via the
// paramgen inclusion validation in the normal Configure path).
func splitSpans(runes []rune, cfg Config) ([]span, error) {
	if len(runes) == 0 {
		return nil, nil
	}
	switch cfg.Strategy {
	case StrategyFixedSize:
		return fixedSizeSpans(runes, cfg.ChunkSize, cfg.Overlap), nil
	case StrategySentence:
		return packSpans(sentenceBoundarySpans(runes, 0, len(runes)), cfg.ChunkSize), nil
	case StrategyRecursive:
		return packSpans(recursiveSpans(runes, cfg.ChunkSize), cfg.ChunkSize), nil
	default:
		return nil, fmt.Errorf("unknown chunking strategy %q", cfg.Strategy)
	}
}

// fixedSizeSpans slides a chunkSize-rune window across runes, advancing by
// chunkSize-overlap runes each step so the trailing overlap runes of one
// chunk repeat at the start of the next. Config.Validate guarantees overlap
// < chunkSize for this strategy, so the window always advances and this
// terminates. The final window is truncated to len(runes) rather than
// padded, so "exact multiple" and "shorter than chunkSize" inputs each
// produce exactly the spans you'd expect: no trailing empty chunk, no
// partial final window when the text divides evenly.
func fixedSizeSpans(runes []rune, chunkSize, overlap int) []span {
	var spans []span
	step := chunkSize - overlap
	for start := 0; start < len(runes); start += step {
		end := min(start+chunkSize, len(runes))
		spans = append(spans, span{start, end})
		if end == len(runes) {
			break
		}
	}
	return spans
}

// isSentenceBoundarySpace reports whether r is whitespace that may follow
// sentence-ending punctuation. Deliberately narrow (space, tab, newline,
// carriage return) rather than unicode.IsSpace's full set — sentence
// boundaries in prose end on ordinary whitespace, and a narrower check
// keeps this heuristic predictable.
func isSentenceBoundarySpace(r rune) bool {
	return r == ' ' || r == '\t' || r == '\n' || r == '\r'
}

// sentenceBoundarySpans splits runes[start:end) on sentence-ending
// punctuation ('.', '!', '?', or a run of them, e.g. "?!") followed by
// whitespace or end-of-text. This is a dependency-free heuristic, not an
// NLP sentence tokenizer (doc.go) — it has no abbreviation handling, so
// "Dr. Smith arrived." splits after "Dr.". Returned spans tile
// [start, end) exactly (each span's end equals the next span's start, the
// last span's end is end) so any contiguous run of them, concatenated, is
// an exact substring of the original document.
func sentenceBoundarySpans(runes []rune, start, end int) []span {
	if start >= end {
		return nil
	}
	var spans []span
	segStart := start
	i := start
	for i < end {
		r := runes[i]
		if r == '.' || r == '!' || r == '?' {
			j := i + 1
			for j < end && (runes[j] == '.' || runes[j] == '!' || runes[j] == '?') {
				j++
			}
			if j >= end || isSentenceBoundarySpace(runes[j]) {
				spans = append(spans, span{segStart, j})
				segStart = j
			}
			i = j
			continue
		}
		i++
	}
	if segStart < end {
		spans = append(spans, span{segStart, end})
	}
	return spans
}

// wordBoundarySpans splits runes[start:end) at the start of each whitespace
// run that follows a non-whitespace rune — i.e. right after each word,
// keeping the trailing whitespace attached to the word that precedes it.
// Like sentenceBoundarySpans, the result tiles [start, end) exactly.
func wordBoundarySpans(runes []rune, start, end int) []span {
	if start >= end {
		return nil
	}
	var spans []span
	segStart := start
	inSpace := isSentenceBoundarySpace(runes[start])
	for i := start + 1; i < end; i++ {
		sp := isSentenceBoundarySpace(runes[i])
		if sp && !inSpace {
			spans = append(spans, span{segStart, i})
			segStart = i
		}
		inSpace = sp
	}
	spans = append(spans, span{segStart, end})
	return spans
}

// paragraphBoundarySpans splits runes[start:end) after each run of two or
// more consecutive '\n' runes (a blank line). Like the other boundary
// functions, the result tiles [start, end) exactly.
func paragraphBoundarySpans(runes []rune, start, end int) []span {
	if start >= end {
		return nil
	}
	var spans []span
	segStart := start
	i := start
	for i < end-1 {
		if runes[i] == '\n' && runes[i+1] == '\n' {
			j := i + 2
			for j < end && runes[j] == '\n' {
				j++
			}
			spans = append(spans, span{segStart, j})
			segStart = j
			i = j
			continue
		}
		i++
	}
	if segStart < end {
		spans = append(spans, span{segStart, end})
	}
	return spans
}

// recursiveSpans implements the "recursive character splitter" shape
// (design doc §3): try paragraph boundaries first; any resulting piece
// still over chunkSize is recursively split on sentence boundaries, then
// word boundaries, then — only if a single word still exceeds chunkSize —
// hard-split at the rune level. Unlike sentenceBoundarySpans alone, this
// guarantees every returned span's length is <= chunkSize (or exactly 1
// rune, if chunkSize is somehow smaller than a single rune can be divided,
// which config.go's gt=0 validation prevents from being <= 0 but doesn't
// prevent being 1).
func recursiveSpans(runes []rune, chunkSize int) []span {
	return splitRegion(runes, 0, len(runes), chunkSize, recursiveBoundaryFuncs)
}

type boundaryFunc func(runes []rune, start, end int) []span

// recursiveBoundaryFuncs is the fixed separator order recursiveSpans tries,
// finest-last: paragraph, then sentence, then word. hardSplitSpans is the
// unconditional fallback once all three are exhausted (span.go's
// splitRegion applies it when boundaryFuncs runs out).
var recursiveBoundaryFuncs = []boundaryFunc{
	paragraphBoundarySpans,
	sentenceBoundarySpans,
	wordBoundarySpans,
}

// splitRegion returns spans tiling [start, end) such that every span's
// length is <= chunkSize. If the region already fits, it's returned
// whole. Otherwise it tries funcs[0]; if that separator doesn't occur in
// this region at all (a single span identical to the input — e.g. no
// blank line in this paragraph-sized region), it falls through to
// funcs[1:] without recursing (recursing on an unchanged span would loop
// forever). Any piece funcs[0] does produce that's still oversized is
// recursively split with the remaining, finer-grained funcs. Once funcs is
// exhausted, hardSplitSpans guarantees termination and the <= chunkSize
// postcondition unconditionally.
func splitRegion(runes []rune, start, end, chunkSize int, funcs []boundaryFunc) []span {
	if end-start <= chunkSize {
		return []span{{start, end}}
	}
	if len(funcs) == 0 {
		return hardSplitSpans(start, end, chunkSize)
	}

	pieces := funcs[0](runes, start, end)
	if len(pieces) <= 1 {
		// This separator found no boundary in the region (or the region
		// was empty) — try the next, finer separator instead of recursing
		// on an unchanged span.
		return splitRegion(runes, start, end, chunkSize, funcs[1:])
	}

	out := make([]span, 0, len(pieces))
	for _, p := range pieces {
		if p.len() > chunkSize {
			out = append(out, splitRegion(runes, p.start, p.end, chunkSize, funcs[1:])...)
		} else {
			out = append(out, p)
		}
	}
	return out
}

// hardSplitSpans splits [start, end) into consecutive chunkSize-rune
// pieces (the final piece truncated, not padded). The fallback of last
// resort for a single "word" (whitespace-free run) longer than chunkSize —
// the only way recursiveSpans can guarantee every span fits.
func hardSplitSpans(start, end, chunkSize int) []span {
	var out []span
	for i := start; i < end; i += chunkSize {
		out = append(out, span{i, min(i+chunkSize, end)})
	}
	return out
}

// packSpans greedily merges consecutive, contiguous units into chunks up
// to chunkSize runes each, without ever splitting a unit. A single unit
// already longer than chunkSize is kept whole as its own oversized chunk
// (this is how sentenceBoundarySpans's output can legitimately exceed
// chunkSize — see doc.go's sentence strategy note); recursiveSpans never
// hands packSpans a unit longer than chunkSize in the first place, so for
// the recursive strategy this path never triggers.
//
// Precondition: units tiles a contiguous range (units[i].end ==
// units[i+1].start for all i) — every boundaryFunc in this file
// guarantees this, so a packed run of units is always a single valid
// substring of the original document.
func packSpans(units []span, chunkSize int) []span {
	if len(units) == 0 {
		return nil
	}
	out := make([]span, 0, len(units))
	cur := units[0]
	for _, u := range units[1:] {
		if u.end-cur.start > chunkSize {
			out = append(out, cur)
			cur = u
			continue
		}
		cur.end = u.end
	}
	out = append(out, cur)
	return out
}
