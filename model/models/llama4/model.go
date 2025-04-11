package llama4

import (
	"bytes"
	"image"
	"slices"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
)

type Model struct {
	model.Base
	model.BytePairEncoding

	*VisionModel `gguf:"v,vision"`
	*Projector   `gguf:"mm"`
	*TextModel
}

type Projector struct {
	Linear1 *nn.Linear `gguf:"linear_1"`
}

func (p *Projector) Forward(ctx ml.Context, visionOutputs ml.Tensor) ml.Tensor {
	return p.Linear1.Forward(ctx, visionOutputs)
}

func New(c fs.Config) (model.Model, error) {
	m := Model{
		BytePairEncoding: model.NewBytePairEncoding(
			c.String("tokenizer.ggml.pretokenizer", `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`),
			&model.Vocabulary{
				Values: c.Strings("tokenizer.ggml.tokens"),
				Types:  c.Uints("tokenizer.ggml.token_type"),
				Merges: c.Strings("tokenizer.ggml.merges"),
				BOS:    int32(c.Uint("tokenizer.ggml.bos_token_id")),
				AddBOS: c.Bool("tokenizer.ggml.add_bos_token", true),
				EOS:    int32(c.Uint("tokenizer.ggml.eos_token_id")),
				AddEOS: c.Bool("tokenizer.ggml.add_eos_token", false),
			},
		),
		VisionModel: newVisionModel(c),
		TextModel:   newTextModel(c),
	}

	m.Cache = kvcache.NewWrapperCache(
		kvcache.NewChunkedAttentionCache(int32(c.Uint("attention.chunk_size")), m.Shift),
		kvcache.NewCausalCache(m.Shift),
	)

	return &m, nil
}

func (m *Model) EncodeMultimodal(ctx ml.Context, multimodalData []byte) (any, error) {
	if len(m.VisionModel.Layers) < 1 {
		return nil, model.ErrNoVisionModel
	}

	img, _, err := image.Decode(bytes.NewReader(multimodalData))
	if err != nil {
		return nil, err
	}

	f32s, aspectRatio, err := m.ProcessImage(ctx, img)
	if err != nil {
		return nil, err
	}

	pixelValues, err := ctx.Input().FromFloatSlice(f32s, len(f32s))
	if err != nil {
		return nil, err
	}

	visionOutputs := m.VisionModel.Forward(ctx, pixelValues)
	return &imageFeature{
		aspectRatio: aspectRatio,
		height:      336,
		width:       336,
		t:           m.Projector.Forward(ctx, visionOutputs),
	}, nil
}

type imageFeature struct {
	aspectRatio   image.Point
	height, width int
	t             ml.Tensor
}

func (m *Model) PostTokenize(inputs []input.Input) ([]input.Input, error) {
	var result []input.Input
	for _, inp := range inputs {
		if inp.Multimodal == nil {
			result = append(result, inp)
		} else {
			image := inp.Multimodal.(*imageFeature)
			patchesPerChunk := image.height / m.patchSize * image.width / m.patchSize

			// TODO
			var imageInputs []input.Input
			imageInputs = append(imageInputs, input.Input{Token: 200080}) /* <|image_start|> */
			if image.aspectRatio.X*image.aspectRatio.Y > 1 {
				for range image.aspectRatio.Y {
					for x := range image.aspectRatio.X {
						imageInputs = append(imageInputs, input.Input{Token: 200092})                                          /* <|patch|> */
						imageInputs = append(imageInputs, slices.Repeat([]input.Input{{Token: 200092}}, patchesPerChunk-1)...) /* <|patch|> */
						if x < image.aspectRatio.X-1 {
							imageInputs = append(imageInputs, input.Input{Token: 200084}) /* <|tile_x_separator|> */
						}
					}

					imageInputs = append(imageInputs, input.Input{Token: 200085}) /* <|tile_y_separator|> */
				}
			}

			imageInputs = append(imageInputs, input.Input{Token: 200090})                                                          /* <|image|> */
			imageInputs = append(imageInputs, input.Input{Token: 200092, Multimodal: image.t, MultimodalHash: inp.MultimodalHash}) /* <|patch|> */
			imageInputs = append(imageInputs, slices.Repeat([]input.Input{{Token: 200092}}, patchesPerChunk-1)...)                 /* <|patch|> */
			imageInputs = append(imageInputs, input.Input{Token: 200081})                                                          /* <|image_end|> */
			result = append(result, imageInputs...)
		}
	}

	return result, nil
}

func (m *Model) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	positions, err := ctx.Input().FromIntSlice(batch.Positions, len(batch.Positions))
	if err != nil {
		return nil, err
	}

	outputs, err := ctx.Input().FromIntSlice(batch.Outputs, len(batch.Outputs))
	if err != nil {
		return nil, err
	}

	return m.TextModel.Forward(ctx, batch.Inputs, positions, outputs, batch, m.Cache), nil
}

func init() {
	model.Register("llama4", New)
}
