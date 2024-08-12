package micrograd

import (
	"fmt"
	"math"
	"slices"
)

// Make micrograd more generic for different data types
type DataType interface {
	int8 | int16 | int64 | float64 | float32
}

// Scalar is one dimentional data
type Scalar[D DataType] struct {
	Data         D
	Grad         D
	Parents      []*Scalar[D]
	BackwardFunc func(*Scalar[D])
	Operation    string
	Label        string
}

// NewScalar returns a scalar with give value (and type)
func NewScalar[D DataType](value D) *Scalar[D] {
	return &Scalar[D]{
		Data: value,
		Grad: 0,
	}
}

// Add adds another scalar, return a new scalar, and add two components to the Parents list
func (s *Scalar[D]) Add(other *Scalar[D]) *Scalar[D] {
	out := &Scalar[D]{
		Data:      s.Data + other.Data,
		Operation: "+",
		Grad:      0,
		Parents:   []*Scalar[D]{s, other},
		BackwardFunc: func(out *Scalar[D]) {
			out.Parents[0].Grad += out.Grad
			out.Parents[1].Grad += out.Grad
		},
	}
	return out
}

func (s *Scalar[D]) Mul(other *Scalar[D]) *Scalar[D] {
	out := &Scalar[D]{
		Data:      s.Data * other.Data,
		Operation: "*",
		Grad:      0,
		Parents:   []*Scalar[D]{s, other},
		BackwardFunc: func(out *Scalar[D]) {
			out.Parents[0].Grad += out.Parents[1].Data * out.Grad
			out.Parents[1].Grad += out.Parents[0].Data * out.Grad
		},
	}
	return out
}

func (s *Scalar[D]) Neg() *Scalar[D] {
	return s.Mul(NewScalar[D](-1))
}

func (s *Scalar[D]) Sub(other *Scalar[D]) *Scalar[D] {
	return s.Add(other.Neg())
}

func (s *Scalar[D]) Div(other *Scalar[D]) *Scalar[D] {
	return s.Mul(other.Pow(-1))
}

func (s *Scalar[D]) Pow(p float64) *Scalar[D] {
	out := &Scalar[D]{
		Data:      D(math.Pow(float64(s.Data), p)),
		Operation: fmt.Sprintf("**%.4f", p),
		Grad:      0,
		Parents:   []*Scalar[D]{s},
		BackwardFunc: func(out *Scalar[D]) {
			out.Parents[0].Grad += D(
				(p * math.Pow(
					float64(s.Data),
					p-1,
				))) * out.Grad
		},
	}
	return out
}

// Backward runs the backward pass through the whole chain, calcuating the grad of all scalars
func (s *Scalar[D]) Backward() {
	topo := []*Scalar[D]{}
	visited := map[*Scalar[D]]struct{}{}
	var build func(v *Scalar[D])
	build = func(v *Scalar[D]) {
		if _, ok := visited[v]; !ok {
			visited[v] = struct{}{}
			for _, parent := range v.Parents {
				build(parent)
			}
			topo = append(topo, v)
		}
	}
	build(s)
	s.Grad = 1
	slices.Reverse(topo)
	for _, v := range topo {
		if v.BackwardFunc != nil {
			v.BackwardFunc(v)
		}
	}
}

// Print show a simple expresstion of how s is made
func (s *Scalar[D]) Print() {
	if s.Parents != nil {
		if len(s.Parents) > 1 {
			fmt.Printf("%s [v:%v g:%v]: %s [v:%v g:%v] %s %s[v:%v g:%v]\n",
				s.Label, s.Data, s.Grad,
				s.Parents[0].Label, s.Parents[0].Data, s.Parents[0].Grad,
				s.Operation,
				s.Parents[1].Label, s.Parents[1].Data, s.Parents[1].Grad,
			)
		}
	}
	for _, p := range s.Parents {
		p.Print()
	}
}
