package micrograd

import (
	"fmt"
	"testing"
)

func TestTrain(t *testing.T) {
	m := NewMLP[float64](3, []int64{4, 4, 1})

	xin := [][]*Scalar[float64]{
		{NewScalar(2.0), NewScalar(3.0), NewScalar(-1.0)},
		{NewScalar(3.0), NewScalar(-1.0), NewScalar(0.5)},
		{NewScalar(0.5), NewScalar(1.0), NewScalar(1.0)},
		{NewScalar(1.0), NewScalar(1.0), NewScalar(-1.0)},
	}

	yout := []*Scalar[float64]{NewScalar(1.0), NewScalar(-1.0), NewScalar(-1.0), NewScalar(1.0)}

	yPredict := []*Scalar[float64]{}

	loss := func(oo, pd []*Scalar[float64]) *Scalar[float64] {
		sum := NewScalar(0.0)
		for i := 0; i < len(oo); i++ {
			sum = sum.Add(oo[i].Sub(pd[i]).Pow(2.0))
		}
		return sum
	}

	// fmt.Println(m.Parameters())
	// m.Apply(xin[0])[0].Print()

	for step := 0; step < 30; step++ {
		yPredict = nil
		for _, x := range xin {
			yPredict = append(yPredict, m.Apply(x)...)
		}
		lossVal := loss(yout, yPredict)

		m.ZeroGrad()
		lossVal.Backward()

		for _, p := range m.Parameters() {
			p.Data += -0.05 * p.Grad
		}

		fmt.Println(step, lossVal.Data)
	}
}
