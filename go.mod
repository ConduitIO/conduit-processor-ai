module github.com/conduitio/conduit-processor-ai

go 1.25.0

require (
	github.com/conduitio/conduit-commons v0.6.0
	github.com/conduitio/conduit-processor-sdk v0.0.0-00010101000000-000000000000
	github.com/jpillora/backoff v1.0.0
	github.com/matryer/is v1.4.1
	go.uber.org/mock v0.6.0
)

require (
	github.com/goccy/go-json v0.10.6 // indirect
	github.com/hamba/avro/v2 v2.28.0 // indirect
	github.com/json-iterator/go v1.1.12 // indirect
	github.com/mattn/go-colorable v0.1.14 // indirect
	github.com/mattn/go-isatty v0.0.20 // indirect
	github.com/mitchellh/mapstructure v1.5.0 // indirect
	github.com/modern-go/concurrent v0.0.0-20180306012644-bacd9c7ef1dd // indirect
	github.com/modern-go/reflect2 v1.0.2 // indirect
	github.com/rs/zerolog v1.35.1 // indirect
	github.com/twmb/go-cache v1.3.0 // indirect
	golang.org/x/exp v0.0.0-20250506013437-ce4c2cf36ca6 // indirect
	golang.org/x/sys v0.33.0 // indirect
	google.golang.org/protobuf v1.36.11 // indirect
)

// TODO(bundle-merge): this replace points at a local, unreleased checkout of
// conduit-processor-sdk's feat/wasm-host-egress branch (the egress package
// this repo depends on isn't tagged yet). It MUST be repointed to a tagged
// conduit-processor-sdk release before this repo's PR merges — do not let
// this ship as-is.
replace github.com/conduitio/conduit-processor-sdk => /private/tmp/claude-501/-Users-devarisbrown-Code-projects-conduit/ca8b4be3-81fb-4e36-9ea0-6d0dd3d91d7a/scratchpad/sdk
