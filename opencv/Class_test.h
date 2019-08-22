#pragma once
class Circle
{
public:
	explicit Circle(double r) : R(r) {}
	explicit Circle(int x, int y = 0) : X(x), Y(y) {}
	explicit Circle(const Circle& c) : R(c.R), X(c.X), Y(c.Y) {}
private:
	double R;
	int    X;
	int    Y;
};