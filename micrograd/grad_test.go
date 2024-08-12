package micrograd

import (
	"fmt"
	"testing"
)

func TestAdd(t *testing.T) {
	a := NewScalar[float32](-4.0)
	b := NewScalar[float32](2.0)
	a.Label = "a"
	b.Label = "b"

	c := a.Add(b)
	c.Label = "c"

	d := a.Mul(b).Add(b.Pow(4))
	c = c.Add(c).Add(NewScalar[float32](1.0))

	c = c.Add(NewScalar[float32](1.0)).Add(c).Add(a)

	d = d.Add(d.Mul(NewScalar[float32](2))).Add(b.Add(a))

	d = d.Add(NewScalar[float32](3).Mul(d)).Add(b.Add(a))

	e := c.Mul(d)
	e.Label = "e"

	f := e.Pow(2)
	f.Label = "f"

	f.Print()

	f.Backward()

	fmt.Println("After backward pass")

	f.Print()
}
