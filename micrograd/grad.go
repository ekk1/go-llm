package micrograd

import (
	"fmt"
	"slices"
)

type DataType interface {
	int8 | int16 | int64 | float64 | float32
}

type Scalar[D DataType] struct {
	Data         D
	Grad         D
	Parents      []*Scalar[D]
	BackwardFunc func(*Scalar[D])
	Operation    string
	Label        string
}

func NewScalar[D DataType](value D) *Scalar[D] {
	return &Scalar[D]{
		Data: value,
		Grad: 0,
	}
}

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

func (s *Scalar[D]) Print() {
	if s.Parents != nil {
		fmt.Printf("%s [v:%v g:%v]: %s [v:%v g:%v] %s %s[v:%v g:%v]\n",
			s.Label, s.Data, s.Grad,
			s.Parents[0].Label, s.Parents[0].Data, s.Parents[0].Grad,
			s.Operation,
			s.Parents[1].Label, s.Parents[1].Data, s.Parents[1].Grad,
		)
	}
	for _, p := range s.Parents {
		p.Print()
	}
}
