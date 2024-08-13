package micrograd

type Module[D DataType] interface {
	ZeroGrad()
	Parameters() []*Scalar[D]
}

type Neuron[D DataType] struct {
	W         []*Scalar[D]
	B         *Scalar[D]
	NonLinear bool
}

func NewNeuron[D DataType](nin int64) *Neuron[D] {
	return nil
}
func (n *Neuron[D]) Apply([]*Scalar[D]) *Scalar[D] {
	return nil
}
func (n *Neuron[D]) Parameters() []*Scalar[D] {
	return nil
}
func (n *Neuron[D]) ZeroGrad() {
}

type Layer[D DataType] struct {
}

func NewLayer[D DataType](nin int64) *Layer[D] {
	return nil
}
func (n *Layer[D]) Apply([]*Scalar[D]) *Scalar[D] {
	return nil
}
func (n *Layer[D]) Parameters() []*Scalar[D] {
	return nil
}
func (n *Layer[D]) ZeroGrad() {
}

type MLP[D DataType] struct {
}

func NewMLP[D DataType](nin int64) *Neuron[D] {
	return nil
}
func (n *MLP[D]) Apply([]*Scalar[D]) *Scalar[D] {
	return nil
}
func (n *MLP[D]) Parameters() []*Scalar[D] {
	return nil
}
func (n *MLP[D]) ZeroGrad() {
}
