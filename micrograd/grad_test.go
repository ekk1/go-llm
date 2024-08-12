package micrograd

import (
	"testing"
)

func TestAdd(t *testing.T) {
	a := NewScalar[float32](3.0)
	b := NewScalar[float32](-1.0)
	c := NewScalar[float32](2.0)
	a.Label = "a"
	b.Label = "b"
	c.Label = "c"

	o := a.Add(a).Add(b).Mul(c)
	o.Label = "o"

	o.Print()

	o.Backward()

	o.Print()
}
