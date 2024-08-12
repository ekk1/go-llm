package micrograd

import (
	"fmt"
	"math"
	"testing"
)

func TestScalar(t *testing.T) {
	a := NewScalar(-4.0)
	b := NewScalar(3.0)
	c := NewScalar(8.0)

	fmt.Println(a.Add(b))
	fmt.Println(b.Add(c))
	fmt.Println(a.Pow(3))
	fmt.Println(b.Pow(2))
	fmt.Println(b.Pow(-1))
	fmt.Println(b.Pow(0.5))
	fmt.Println(NewScalar(2.0).Pow(0.5))
	fmt.Println(a.Mul(b))
	fmt.Println(a.Mul(c))
	fmt.Println(a.Sub(b))
	fmt.Println(b.Sub(c))
	fmt.Println(c.Div(a))
	fmt.Println(c.Div(b))
	fmt.Println(b.Neg())

	fmt.Println(a.Tanh())
	fmt.Println(math.Tanh(-4.0))
	fmt.Println(b.Tanh())
	fmt.Println(math.Tanh(3.0))
	fmt.Println(c.Tanh())
	fmt.Println(math.Tanh(8.0))

	o := a.Tanh()
	o.Backward()
	fmt.Println(a.Grad)
}

func TestBackward(t *testing.T) {
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

	// f.Print()

	f.Backward()

	fmt.Println("After backward pass")

	// f.Print()
}
