package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"regexp"
	"slices"
	"strings"
	"text/template/parse"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/template"
	"github.com/ollama/ollama/types/model"
)

var intermediateBlobs map[string]string = make(map[string]string)

type layerGGML struct {
	Layer
	*ggml.GGML
}

func parseFromModel(ctx context.Context, name model.Name, fn func(api.ProgressResponse)) (layers []*layerGGML, err error) {
	m, err := ParseNamedManifest(name)
	switch {
	case errors.Is(err, os.ErrNotExist):
		if err := PullModel(ctx, name.String(), &registryOptions{}, fn); err != nil {
			return nil, err
		}

		m, err = ParseNamedManifest(name)
		if err != nil {
			return nil, err
		}
	case err != nil:
		return nil, err
	}

	for _, layer := range m.Layers {
		layer, err := NewLayerFromLayer(layer.Digest, layer.MediaType, name.DisplayShortest())
		if err != nil {
			return nil, err
		}

		switch layer.MediaType {
		case "application/vnd.ollama.image.model",
			"application/vnd.ollama.image.projector",
			"application/vnd.ollama.image.adapter":
			blobpath, err := GetBlobsPath(layer.Digest)
			if err != nil {
				return nil, err
			}

			blob, err := os.Open(blobpath)
			if err != nil {
				return nil, err
			}
			defer blob.Close()

			f, _, err := ggml.Decode(blob, 0)
			if err != nil {
				return nil, err
			}

			layers = append(layers, &layerGGML{layer, f})
		default:
			layers = append(layers, &layerGGML{layer, nil})
		}
	}

	return layers, nil
}

func detectChatTemplate(layers []*layerGGML) ([]*layerGGML, error) {
	for _, layer := range layers {
		if s := layer.GGML.KV().ChatTemplate(); s != "" {
			if t, err := template.Named(s); err != nil {
				slog.Debug("template detection", "error", err, "template", s)
			} else {
				layer, err := NewLayer(t.Reader(), "application/vnd.ollama.image.template")
				if err != nil {
					return nil, err
				}

				layer.status = fmt.Sprintf("using autodetected template %s", t.Name)
				layers = append(layers, &layerGGML{layer, nil})

				if t.Parameters != nil {
					var b bytes.Buffer
					if err := json.NewEncoder(&b).Encode(t.Parameters); err != nil {
						return nil, err
					}

					layer, err := NewLayer(&b, "application/vnd.ollama.image.params")
					if err != nil {
						return nil, err
					}

					layers = append(layers, &layerGGML{layer, nil})
				}
			}
		}
	}

	return layers, nil
}

func detectContentType(r io.Reader) (string, error) {
	var b bytes.Buffer
	if _, err := io.Copy(&b, r); err != nil {
		return "", err
	}

	if contentType := ggml.DetectContentType(b.Bytes()); contentType != "" {
		return contentType, nil
	}

	if contentType := http.DetectContentType(b.Bytes()); contentType != "application/octet-stream" {
		return contentType, nil
	}

	return "unknown", nil
}

func parseObjects(s string) []map[string]any {
	var objs []map[string]any
	for offset := 0; offset < len(s); {
		var obj map[string]any
		decoder := json.NewDecoder(strings.NewReader(s[offset:]))
		if err := decoder.Decode(&obj); errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
			break
		} else if syntax := &(json.SyntaxError{}); errors.As(err, &syntax) {
			// skip over any syntax errors
			offset += int(syntax.Offset)
		} else if unmarshalType := &(json.UnmarshalTypeError{}); errors.As(err, &unmarshalType) {
			// skip over any unmarshalable types
			offset += int(unmarshalType.Offset)
		} else if err != nil {
			return nil
		} else {
			offset += int(decoder.InputOffset())
			objs = append(objs, obj)
		}
	}

	return objs
}

// parseToolCalls attempts to parse a JSON string into a slice of ToolCalls.
// mxyng: this only really works if the input contains tool calls in some JSON format
func (m *Model) parseToolCalls(s string) ([]api.ToolCall, bool) {
	// create a subtree from the node that ranges over .ToolCalls
	tmpl := m.Template.Subtree(func(n parse.Node) bool {
		if t, ok := n.(*parse.RangeNode); ok {
			return slices.Contains(template.Identifiers(t.Pipe), "ToolCalls")
		}

		return false
	})

	if tmpl == nil {
		slog.Debug("parseToolCalls: no ToolCalls template found")
		return nil, false
	}

	slog.Debug("parseToolCalls: executing template with test data", "input", s)

	var b bytes.Buffer
	if err := tmpl.Execute(&b, map[string][]api.ToolCall{
		"ToolCalls": {
			{
				Function: api.ToolCallFunction{
					Name: "@@name@@",
					Arguments: api.ToolCallFunctionArguments{
						"@@argument@@": 1,
					},
				},
			},
		},
	}); err != nil {
		slog.Debug("parseToolCalls: template execution failed", "error", err)
		return nil, false
	}

	slog.Debug("parseToolCalls: template executed successfully", "output", b.String())

	templateObjects := parseObjects(b.String())
	if len(templateObjects) == 0 {
		return nil, false
	}

	slog.Debug("parseToolCalls: template objects", "objects", templateObjects)

	// find the keys that correspond to the name and arguments fields
	var name, arguments string
	for k, v := range templateObjects[0] {
		switch v.(type) {
		case string:
			name = k
		case map[string]any:
			arguments = k
		}
	}

	if name == "" || arguments == "" {
		return nil, false
	}

	responseObjects := parseObjects(s)
	if len(responseObjects) == 0 {
		return nil, false
	}

	// collect all nested objects
	var collect func(any) []map[string]any
	collect = func(obj any) (all []map[string]any) {
		switch o := obj.(type) {
		case map[string]any:
			all = append(all, o)
			for _, v := range o {
				all = append(all, collect(v)...)
			}
		case []any:
			for _, v := range o {
				all = append(all, collect(v)...)
			}
		}

		return all
	}

	var objs []map[string]any
	for _, p := range responseObjects {
		objs = append(objs, collect(p)...)
	}

	var toolCalls []api.ToolCall
	for _, kv := range objs {
		n, nok := kv[name].(string)
		a, aok := kv[arguments].(map[string]any)
		if nok && aok {
			toolCalls = append(toolCalls, api.ToolCall{
				Function: api.ToolCallFunction{
					Name:      n,
					Arguments: a,
				},
			})
		}
	}

	return toolCalls, len(toolCalls) > 0
}

func (m *Model) ParseToolCallsNew(s string) ([]api.ToolCall, bool) {
	// Parse both Python function calls and JSON function calls into ToolCall structs
	// Example inputs:
	// Python: func(a=2, b=2)
	// JSON: {"function": {"name": "func", "arguments": {"a": 2, "b": 2}}}
	// JSON array: [{"name": "func", "arguments": {"a": 2}}]

	slog.Debug("parsing function calls", "input", s)

	// Trim whitespace from input
	s = strings.TrimSpace(s)

	// Try JSON array parsing first if input starts with [
	if strings.HasPrefix(s, "[") {
		var jsonArray []map[string]any
		decoder := json.NewDecoder(strings.NewReader(s))
		if err := decoder.Decode(&jsonArray); err == nil {
			// Ensure there's no trailing content after the array
			var dummy any
			if !decoder.More() && decoder.Decode(&dummy) != nil {
				var toolCalls []api.ToolCall
				for _, obj := range jsonArray {
					if calls, ok := parseJSONToolCalls(obj); ok {
						toolCalls = append(toolCalls, calls...)
					}
				}

				// Only return success if we found valid tool calls
				if len(toolCalls) > 0 {
					// Check if any of the tool calls are malformed
					for _, call := range toolCalls {
						if call.Function.Name == "" || len(call.Function.Arguments) == 0 {
							return nil, false
						}
					}
					return toolCalls, true
				}
			}
		}
	}

	// Try parsing single objects and Python function calls
	var toolCalls []api.ToolCall
	for offset := 0; offset < len(s); {
		// Try single object if not array
		var jsonObj map[string]any
		decoder := json.NewDecoder(strings.NewReader(s[offset:]))
		if err := decoder.Decode(&jsonObj); err != nil {
			// If we can't parse JSON, try Python function call
			re := regexp.MustCompile(`(\w+)\((.*?)\)`)
			if match := re.FindStringSubmatchIndex(s[offset:]); match != nil {
				// Found a Python function call
				name := s[offset+match[2] : offset+match[3]]
				args := s[offset+match[4] : offset+match[5]]

				arguments := make(api.ToolCallFunctionArguments)
				if strings.Contains(args, "=") { // Keyword args
					pairs := strings.Split(args, ",")
					for _, pair := range pairs {
						pair = strings.TrimSpace(pair)
						kv := strings.Split(pair, "=")
						if len(kv) == 2 {
							key := strings.TrimSpace(kv[0])
							value := strings.TrimSpace(kv[1])
							arguments[key] = value
						}
					}
					toolCalls = append(toolCalls, api.ToolCall{
						Function: api.ToolCallFunction{
							Name:      name,
							Arguments: arguments,
						},
					})
				}
				// Skip past the function call
				offset += match[1]
			} else {
				// No JSON or Python function call found, move forward
				offset++
			}
			continue
		}
		// Successfully parsed object, process it
		if calls, ok := parseJSONToolCalls(jsonObj); ok {
			toolCalls = append(toolCalls, calls...)
		}
		offset += int(decoder.InputOffset())
	}

	// Only return success if we found valid tool calls and no errors
	if len(toolCalls) > 0 {
		// Check if any of the tool calls are malformed
		for _, call := range toolCalls {
			if call.Function.Name == "" || len(call.Function.Arguments) == 0 {
				return nil, false
			}
		}
		return toolCalls, true
	}

	return nil, false
}

// ToolCallFormat represents different possible formats for tool calls
type toolCallFormat struct {
	// Direct format
	Name      string         `json:"name,omitempty"`
	Arguments map[string]any `json:"arguments,omitempty"`

	// Command-r-plus format
	ToolName   string         `json:"tool_name,omitempty"`
	Parameters map[string]any `json:"parameters,omitempty"`

	// Function format
	Function *struct {
		Name       string         `json:"name"`
		Arguments  map[string]any `json:"arguments,omitempty"`
		Parameters map[string]any `json:"parameters,omitempty"`
	} `json:"function,omitempty"`

	// Xlam format
	ToolCalls []toolCallFormat `json:"tool_calls,omitempty"`
}

func parseJSONToolCalls(obj map[string]any) ([]api.ToolCall, bool) {
	// Helper to convert any to []any safely
	toArray := func(v any) []any {
		if arr, ok := v.([]any); ok {
			return arr
		}
		return nil
	}

	// Convert a single format to a tool call
	makeToolCall := func(f toolCallFormat) (api.ToolCall, bool) {
		switch {
		case f.Name != "" && f.Arguments != nil:
			return api.ToolCall{
				Function: api.ToolCallFunction{
					Name:      f.Name,
					Arguments: f.Arguments,
				},
			}, true
		case f.ToolName != "" && f.Parameters != nil:
			return api.ToolCall{
				Function: api.ToolCallFunction{
					Name:      f.ToolName,
					Arguments: f.Parameters,
				},
			}, true
		case f.Function != nil && f.Function.Name != "":
			args := f.Function.Arguments
			if args == nil {
				args = f.Function.Parameters
			}
			if args != nil {
				return api.ToolCall{
					Function: api.ToolCallFunction{
						Name:      f.Function.Name,
						Arguments: args,
					},
				}, true
			}
		}
		return api.ToolCall{}, false
	}

	// Try parsing as array first
	if arr := toArray(obj); arr != nil {
		var calls []api.ToolCall
		for _, item := range arr {
			if itemMap, ok := item.(map[string]any); ok {
				var format toolCallFormat
				data, _ := json.Marshal(itemMap)
				if err := json.Unmarshal(data, &format); err == nil {
					if call, ok := makeToolCall(format); ok {
						calls = append(calls, call)
					}
				}
			}
		}
		if len(calls) > 0 {
			return calls, true
		}
	}

	// Try parsing as single object
	var format toolCallFormat
	data, _ := json.Marshal(obj)
	if err := json.Unmarshal(data, &format); err != nil {
		return nil, false
	}

	// Handle xlam format (tool_calls array)
	if len(format.ToolCalls) > 0 {
		var calls []api.ToolCall
		for _, f := range format.ToolCalls {
			if call, ok := makeToolCall(f); ok {
				calls = append(calls, call)
			}
		}
		if len(calls) > 0 {
			return calls, true
		}
	}

	// Try as single tool call
	if call, ok := makeToolCall(format); ok {
		return []api.ToolCall{call}, true
	}

	return nil, false
}
