package micrograd

import "math/rand"

type Module[D DataType] interface {
	ZeroGrad()
	Parameters() []*Scalar[D]
}

type Neuron[D DataType] struct {
	W             []*Scalar[D]
	B             *Scalar[D]
	NonLinearFunc string
}

func NewNeuron[D DataType](nin int64) *Neuron[D] {
	n := &Neuron[D]{}
	n.B = NewScalar[D](0.0)
	for i := int64(0); i < nin; i++ {
		// all w in n.W will be [-1, 1]
		n.W = append(n.W, NewScalar(D(-1.0+rand.Float64()*2.0)))
	}
	return n
}
func (n *Neuron[D]) Apply(input []*Scalar[D]) *Scalar[D] {
	minDim := len(n.W)
	if len(input) < minDim {
		minDim = len(input)
	}
	firstSum := NewScalar[D](0)
	for i := 0; i < minDim; i++ {
		firstSum = firstSum.Add(input[i].Mul(n.W[i]))
	}
	outScalar := firstSum.Add(n.B)
	switch n.NonLinearFunc {
	case "relu":
		return outScalar.ReLU()
	case "tanh":
		return outScalar.Tanh()
	default:
		return outScalar
	}
}
func (n *Neuron[D]) Parameters() []*Scalar[D] {
	return append(n.W, n.B)
}
func (n *Neuron[D]) ZeroGrad() {
	for _, v := range n.Parameters() {
		v.Grad = 0
	}
}

type Layer[D DataType] struct {
	Neurons []*Neuron[D]
}

func NewLayer[D DataType](nin, nout int64) *Layer[D] {
	l := &Layer[D]{Neurons: []*Neuron[D]{}}
	for i := int64(0); i < nout; i++ {
		l.Neurons = append(l.Neurons, NewNeuron[D](nin))
	}
	return l
}
func (l *Layer[D]) Apply(input []*Scalar[D]) []*Scalar[D] {
	out := []*Scalar[D]{}
	for _, n := range l.Neurons {
		nx := n.Apply(input)
		out = append(out, nx)
	}
	return out
}
func (l *Layer[D]) Parameters() []*Scalar[D] {
	out := []*Scalar[D]{}
	for _, n := range l.Neurons {
		nx := n.Parameters()
		out = append(out, nx...)
	}
	return out
}
func (l *Layer[D]) ZeroGrad() {
	for _, n := range l.Neurons {
		n.ZeroGrad()
	}
}
func (l *Layer[D]) SetNonLinear(f string) {
	for _, n := range l.Neurons {
		n.NonLinearFunc = f
	}
}

type MLP[D DataType] struct {
	Layers []*Layer[D]
}

func NewMLP[D DataType](nin int64, layerOut []int64) *MLP[D] {
	m := &MLP[D]{Layers: []*Layer[D]{}}
	sizeList := []int64{nin}
	sizeList = append(sizeList, layerOut...)
	for i := 0; i < len(layerOut); i++ {
		l := NewLayer[D](sizeList[i], sizeList[i+1])
		if i != len(layerOut)-1 {
			l.SetNonLinear("tanh")
		}
		m.Layers = append(m.Layers, l)
	}
	return m
}
func (m *MLP[D]) Apply(input []*Scalar[D]) []*Scalar[D] {
	out := input
	for _, l := range m.Layers {
		out = l.Apply(out)
	}
	return out
}
func (m *MLP[D]) Parameters() []*Scalar[D] {
	out := []*Scalar[D]{}
	for _, l := range m.Layers {
		nx := l.Parameters()
		out = append(out, nx...)
	}
	return out
}
func (m *MLP[D]) ZeroGrad() {
	for _, l := range m.Layers {
		l.ZeroGrad()
	}
}
