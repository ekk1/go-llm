package micrograd

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
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

func ReadMNISTImage(p string) [][][]byte {
	imageList := [][][]byte{}
	f, err := os.Open(p)
	defer f.Close()
	if err != nil {
		panic(err)
	}
	type Header struct {
		Magic uint32
		Size  uint32
		Rows  uint32
		Cols  uint32
	}
	h := &Header{}
	binary.Read(f, binary.BigEndian, h)
	if h.Magic != 2051 {
		panic("Wrong magic for image")
	}
	for p := 0; p < int(h.Size); p++ {
		imgTemp := [][]byte{}
		for i := 0; i < int(h.Cols); i++ {
			rowData := make([]byte, h.Rows)
			f.Read(rowData)
			imgTemp = append(imgTemp, rowData)
		}
		imageList = append(imageList, imgTemp)
	}
	return imageList
}

func ReadMNISTLabel(p string) []byte {
	labelList := []byte{}
	f, err := os.Open(p)
	defer f.Close()
	if err != nil {
		panic(err)
	}
	type Header struct {
		Magic uint32
		Size  uint32
	}
	h := &Header{}
	binary.Read(f, binary.BigEndian, h)
	if h.Magic != 2049 {
		panic("Wrong magic for image")
	}
	for p := 0; p < int(h.Size); p++ {
		labelData := make([]byte, 1)
		f.Read(labelData)
		labelList = append(labelList, labelData[0])
	}
	return labelList
}

func TestMNIST(t *testing.T) {
	train_data_path := "../mnist/train-images.idx3-ubyte"
	train_label_path := "../mnist/train-labels.idx1-ubyte"
	imageList := ReadMNISTImage(train_data_path)
	labelList := ReadMNISTLabel(train_label_path)
	if len(imageList) != len(labelList) {
		panic("Mismatch")
	}
	//fmt.Println(labelList[0])
	m := NewMLP[float32](784, []int64{512, 384, 10})
	//fmt.Println(m.Layers[0].Neurons[0])
	yPredict := []*Scalar[float32]{}
	loss := func(oo, pd []*Scalar[float32]) *Scalar[float32] {
		sum := NewScalar[float32](0.0)
		epsilon := float32(1e-12)
		for i := 0; i < len(oo); i++ {
			target := oo[i].Data
			prediction := pd[i].Data
			// Add a small epsilon to prevent log(0)
			sum = sum.Add(NewScalar(target * float32(math.Log(float64(prediction+epsilon)))))
		}
		return sum
	}
	//for i := 0; i < len(imageList); i++ {
	for i := 0; i < 3; i++ {
		yTarget := make([]*Scalar[float32], 10)
		for n := 0; n < 10; n++ {
			yTarget[n] = NewScalar[float32](0)
		}
		yTarget[int(labelList[i])] = NewScalar[float32](1)

		inputData := []*Scalar[float32]{}
		for _, row := range imageList[i] {
			for _, col := range row {
				inputData = append(inputData, NewScalar(float32(col)/255.0))
			}
		}

		for step := 0; step < 30; step++ {
			yPredict = nil
			yPredict = m.Apply(inputData)
			for i := 0; i < len(yTarget); i++ {
				fmt.Println(yTarget[i].Data, yPredict[i].Data)
			}
			lossVal := loss(yTarget, yPredict)

			m.ZeroGrad()
			lossVal.Backward()

			//lossVal.Print()

			for _, p := range m.Parameters() {
				//if p.Grad > 1.0e3 {
				//p.Grad = 1.0e3
				//}
				//if p.Grad < -1.0e3 {
				//p.Grad = -1.0e3
				//}
				p.Data += -0.05 * p.Grad
			}
			fmt.Println("Step", step, lossVal.Data)
		}
	}
}
