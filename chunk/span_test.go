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

import (
	"strings"
	"testing"

	"github.com/matryer/is"
)

// chunkTexts slices runes at each span and returns the resulting strings,
// the shape most tests want to assert against.
func chunkTexts(runes []rune, spans []span) []string {
	out := make([]string, len(spans))
	for i, sp := range spans {
		out[i] = string(runes[sp.start:sp.end])
	}
	return out
}

// --- fixed_size ---

func TestFixedSizeSpans_NoOverlap(t *testing.T) {
	is := is.New(t)
	runes := []rune(strings.Repeat("a", 25))

	spans := fixedSizeSpans(runes, 10, 0)

	is.Equal(len(spans), 3)
	is.Equal(spans[0], span{0, 10})
	is.Equal(spans[1], span{10, 20})
	is.Equal(spans[2], span{20, 25})
}

func TestFixedSizeSpans_ExactMultiple(t *testing.T) {
	is := is.New(t)
	runes := []rune(strings.Repeat("a", 20))

	spans := fixedSizeSpans(runes, 10, 0)

	// No trailing empty/zero-length chunk when the text divides evenly.
	is.Equal(len(spans), 2)
	is.Equal(spans[0], span{0, 10})
	is.Equal(spans[1], span{10, 20})
}

func TestFixedSizeSpans_ShorterThanChunkSize(t *testing.T) {
	is := is.New(t)
	runes := []rune("hello")

	spans := fixedSizeSpans(runes, 10, 0)

	is.Equal(len(spans), 1)
	is.Equal(spans[0], span{0, 5})
}

func TestFixedSizeSpans_Empty(t *testing.T) {
	is := is.New(t)
	spans := fixedSizeSpans(nil, 10, 0)
	is.Equal(len(spans), 0)
}

func TestFixedSizeSpans_WithOverlap(t *testing.T) {
	is := is.New(t)
	runes := []rune(strings.Repeat("a", 25))

	spans := fixedSizeSpans(runes, 10, 3)

	is.Equal(spans, []span{{0, 10}, {7, 17}, {14, 24}, {21, 25}})
	// Each chunk after the first starts 3 runes before the previous one
	// ended - the configured overlap, verified directly rather than just
	// via the raw offsets above.
	for i := 1; i < len(spans); i++ {
		is.Equal(spans[i-1].end-spans[i].start, 3)
	}
}

func TestFixedSizeSpans_OverlapNeverStalls(t *testing.T) {
	// Regression guard for the case Config.Validate exists to prevent:
	// if overlap were allowed to equal chunkSize, step would be 0 and this
	// would loop forever. This test documents *why* Validate requires
	// overlap < chunkSize by exercising the boundary value directly
	// (overlap = chunkSize-1, the largest legal value) and asserting
	// forward progress and termination.
	is := is.New(t)
	runes := []rune(strings.Repeat("a", 50))

	spans := fixedSizeSpans(runes, 10, 9)

	is.True(len(spans) > 1)
	is.Equal(spans[len(spans)-1].end, 50)
}

// --- sentence boundaries ---

func TestSentenceBoundarySpans_Basic(t *testing.T) {
	is := is.New(t)
	text := "Hello world. How are you? Fine!"
	runes := []rune(text)

	spans := sentenceBoundarySpans(runes, 0, len(runes))
	texts := chunkTexts(runes, spans)

	// Boundaries fall right after the terminating punctuation; the space
	// that follows it is attached to the *next* sentence's span, not the
	// one that just ended (see sentenceBoundarySpans's doc comment).
	is.Equal(texts, []string{"Hello world.", " How are you?", " Fine!"})
	is.Equal(strings.Join(texts, ""), text) // still an exact reconstruction
}

func TestSentenceBoundarySpans_NoPunctuation(t *testing.T) {
	is := is.New(t)
	runes := []rune("no sentence enders here")

	spans := sentenceBoundarySpans(runes, 0, len(runes))

	// No boundary found: the whole region comes back as one span, the
	// signal splitRegion uses to fall through to the next separator.
	is.Equal(len(spans), 1)
	is.Equal(spans[0], span{0, len(runes)})
}

func TestSentenceBoundarySpans_TilesContiguously(t *testing.T) {
	is := is.New(t)
	text := "One. Two? Three! Four."
	runes := []rune(text)

	spans := sentenceBoundarySpans(runes, 0, len(runes))

	is.True(len(spans) > 1)
	for i := 1; i < len(spans); i++ {
		is.Equal(spans[i-1].end, spans[i].start) // tiles with no gap/overlap
	}
	is.Equal(spans[0].start, 0)
	is.Equal(spans[len(spans)-1].end, len(runes))
}

// --- word boundaries ---

func TestWordBoundarySpans_TilesContiguously(t *testing.T) {
	is := is.New(t)
	text := "the quick brown fox jumps"
	runes := []rune(text)

	spans := wordBoundarySpans(runes, 0, len(runes))
	texts := chunkTexts(runes, spans)

	is.Equal(strings.Join(texts, ""), text) // exact reconstruction
	is.Equal(len(spans), 5)
}

func TestWordBoundarySpans_NoWhitespace(t *testing.T) {
	is := is.New(t)
	runes := []rune("onelongword")

	spans := wordBoundarySpans(runes, 0, len(runes))

	is.Equal(len(spans), 1)
}

// --- paragraph boundaries ---

func TestParagraphBoundarySpans_Basic(t *testing.T) {
	is := is.New(t)
	text := "First paragraph.\n\nSecond paragraph.\n\nThird."
	runes := []rune(text)

	spans := paragraphBoundarySpans(runes, 0, len(runes))
	texts := chunkTexts(runes, spans)

	is.Equal(texts, []string{"First paragraph.\n\n", "Second paragraph.\n\n", "Third."})
}

// --- recursive: guarantees every span fits, unlike sentence alone ---

func TestRecursiveSpans_EveryChunkFits(t *testing.T) {
	is := is.New(t)
	text := "First paragraph with several words in it.\n\n" +
		"Second paragraph is a good bit longer than the first one and has multiple sentences. " +
		"Here is another sentence to pad it out further so it exceeds the chunk size on its own.\n\n" +
		"Third, short."
	runes := []rune(text)
	const chunkSize = 40

	spans := recursiveSpans(runes, chunkSize)

	is.True(len(spans) > 1)
	for _, sp := range spans {
		is.True(sp.len() <= chunkSize)
	}
	// Tiles contiguously and covers the whole document, same contract as
	// every other boundary function.
	is.Equal(spans[0].start, 0)
	is.Equal(spans[len(spans)-1].end, len(runes))
	for i := 1; i < len(spans); i++ {
		is.Equal(spans[i-1].end, spans[i].start)
	}
}

func TestRecursiveSpans_PrefersParagraphBoundary(t *testing.T) {
	is := is.New(t)
	// Each paragraph individually fits chunkSize, but the whole text
	// doesn't - recursive should split at the paragraph boundary, not
	// fragment down to sentence/word level.
	para1 := "Short first paragraph."
	para2 := "Short second paragraph."
	text := para1 + "\n\n" + para2
	runes := []rune(text)
	const chunkSize = 30 // bigger than either paragraph, smaller than both combined

	spans := recursiveSpans(runes, chunkSize)
	texts := chunkTexts(runes, spans)

	is.Equal(texts, []string{para1 + "\n\n", para2})
}

func TestRecursiveSpans_HardSplitFallback(t *testing.T) {
	is := is.New(t)
	// A single "word" (no whitespace at all) longer than chunkSize can't
	// be split by paragraph, sentence, or word boundaries - only the hard
	// rune-level fallback can make it fit.
	runes := []rune(strings.Repeat("x", 25))
	const chunkSize = 10

	spans := recursiveSpans(runes, chunkSize)

	is.Equal(spans, []span{{0, 10}, {10, 20}, {20, 25}})
}

// --- packSpans ---

func TestPackSpans_MergesUpToChunkSize(t *testing.T) {
	is := is.New(t)
	units := []span{{0, 5}, {5, 8}, {8, 12}, {12, 13}}

	packed := packSpans(units, 10)

	// {0,5}+{5,8} merges (span so far is 8 runes, <=10); adding {8,12}
	// would make it 12 (>10), so {0,8} is flushed and {8,12} starts a new
	// group; {8,12}+{12,13} merges (5 runes, <=10) into the final {8,13}.
	is.Equal(packed, []span{{0, 8}, {8, 13}})
}

func TestPackSpans_KeepsOversizedUnitWhole(t *testing.T) {
	is := is.New(t)
	units := []span{{0, 15}, {15, 18}} // first unit alone exceeds chunkSize=10

	packed := packSpans(units, 10)

	is.Equal(packed, []span{{0, 15}, {15, 18}})
}

func TestPackSpans_Empty(t *testing.T) {
	is := is.New(t)
	is.Equal(len(packSpans(nil, 10)), 0)
}

// --- unicode / multibyte correctness ---

func TestFixedSizeSpans_UnicodeOffsetsAreRuneExact(t *testing.T) {
	is := is.New(t)
	// Mix of multi-byte runes: accented Latin, CJK, and an emoji (which
	// itself is outside the BMP - a byte-offset scheme would either split
	// a UTF-8 continuation byte or a surrogate-style boundary here).
	//nolint:gosmopolitan // intentional non-Latin test fixture: this is exactly what the unicode-offset test needs to exercise
	text := "café 日本語 emoji: 🎉🎉 done"
	runes := []rune(text)
	is.True(len(runes) < len([]byte(text))) // proves the text is multi-byte

	spans := fixedSizeSpans(runes, 5, 0)

	var rebuilt strings.Builder
	for _, sp := range spans {
		chunk := string(runes[sp.start:sp.end])
		is.True(len(chunk) > 0)
		// Every chunk must itself be valid UTF-8 text with no
		// replacement/mangled runes - round-tripping through []rune and
		// back to string can't corrupt it, but this asserts the property
		// directly rather than trusting the mechanism.
		for _, r := range chunk {
			is.True(r != '�')
		}
		rebuilt.WriteString(chunk)
	}
	is.Equal(rebuilt.String(), text) // exact reconstruction, byte for byte
}

func TestRecursiveSpans_UnicodeChunksFitByRuneCountNotByteCount(t *testing.T) {
	is := is.New(t)
	//nolint:gosmopolitan // intentional CJK test fixture: proves offsets/chunkSize are rune-, not byte-, counted
	text := strings.Repeat("日本語テスト ", 10) // each rune is 3 bytes in UTF-8
	runes := []rune(text)
	const chunkSize = 15

	spans := packSpans(recursiveSpans(runes, chunkSize), chunkSize)

	is.True(len(spans) > 1)
	sawMultiByteChunk := false
	for _, sp := range spans {
		is.True(sp.len() <= chunkSize) // bounded in runes, the contract this strategy guarantees
		chunkBytes := len(string(runes[sp.start:sp.end]))
		if chunkBytes > sp.len() {
			sawMultiByteChunk = true
		}
	}
	// At least one packed chunk actually contains the multi-byte CJK text
	// (as opposed to only ever landing on a lone ASCII space fragment),
	// proving the rune-count bound is doing real work here, not
	// accidentally passing because every chunk happened to be ASCII.
	is.True(sawMultiByteChunk)
}
