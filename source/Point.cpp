/* Point.cpp
Copyright (c) 2014 by Michael Zahniser

Endless Sky is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later version.

Endless Sky is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program. If not, see <https://www.gnu.org/licenses/>.
*/

#include "Point.h"

#ifndef __SSE3__
#include <algorithm>
#include <cmath>
using namespace std;
#endif



Point::Point() noexcept
#ifdef __SSE3__
	: v(_mm_setzero_pd())
#else
	: x(0.), y(0.), z(0.)
#endif
{
#ifdef __SSE3__
	val.z = 0.;
#endif
}



Point::Point(double x, double y, double z) noexcept
#ifdef __SSE3__
	: v(_mm_set_pd(y, x))
#else
	: x(x), y(y), z(z)
#endif
{
#ifdef __SSE3__
	val.z = z;
#endif
}



// Check if the point is anything but (0, 0, 0).
Point::operator bool() const noexcept
{
	return !!*this;
}



bool Point::operator!() const noexcept
{
#ifdef __SSE3__
	return (!val.x & !val.y & !val.z);
#else
	return (!x & !y & !z);
#endif
}



Point Point::operator+(const Point &point) const
{
#ifdef __SSE3__
	Point result(v + point.v);
	result.val.z = val.z + point.val.z;
	return result;
#else
	return Point(x + point.x, y + point.y, z + point.z);
#endif
}



Point &Point::operator+=(const Point &point)
{
#ifdef __SSE3__
	v += point.v;
	val.z += point.val.z;
#else
	x += point.x;
	y += point.y;
	z += point.z;
#endif
	return *this;
}



Point Point::operator-(const Point &point) const
{
#ifdef __SSE3__
	Point result(v - point.v);
	result.val.z = val.z - point.val.z;
	return result;
#else
	return Point(x - point.x, y - point.y, z - point.z);
#endif
}



Point &Point::operator-=(const Point &point)
{
#ifdef __SSE3__
	v -= point.v;
	val.z -= point.val.z;
#else
	x -= point.x;
	y -= point.y;
	z -= point.z;
#endif
	return *this;
}



Point Point::operator-() const
{
	return Point() - *this;
}



Point Point::operator*(double scalar) const
{
#ifdef __SSE3__
	Point result(v * _mm_loaddup_pd(&scalar));
	result.val.z = val.z * scalar;
	return result;
#else
	return Point(x * scalar, y * scalar, z * scalar);
#endif
}



Point operator*(double scalar, const Point &point)
{
#ifdef __SSE3__
	Point result(point.v * _mm_loaddup_pd(&scalar));
	result.val.z = point.val.z * scalar;
	return result;
#else
	return Point(point.x * scalar, point.y * scalar, point.z * scalar);
#endif
}



Point &Point::operator*=(double scalar)
{
#ifdef __SSE3__
	v *= _mm_loaddup_pd(&scalar);
	val.z *= scalar;
#else
	x *= scalar;
	y *= scalar;
	z *= scalar;
#endif
	return *this;
}



Point Point::operator*(const Point &other) const
{
#ifdef __SSE3__
	Point result;
	result.v = v * other.v;
	result.val.z = val.z * other.val.z;
	return result;
#else
	return Point(x * other.x, y * other.y, z * other.z);
#endif
}



Point &Point::operator*=(const Point &other)
{
#ifdef __SSE3__
	v *= other.v;
	val.z *= other.val.z;
#else
	x *= other.x;
	y *= other.y;
	z *= other.z;
#endif
	return *this;
}



Point Point::operator/(double scalar) const
{
#ifdef __SSE3__
	Point result(v / _mm_loaddup_pd(&scalar));
	result.val.z = val.z / scalar;
	return result;
#else
	return Point(x / scalar, y / scalar, z / scalar);
#endif
}



Point &Point::operator/=(double scalar)
{
#ifdef __SSE3__
	v /= _mm_loaddup_pd(&scalar);
	val.z /= scalar;
#else
	x /= scalar;
	y /= scalar;
	z /= scalar;
#endif
	return *this;
}



void Point::Set(double x, double y, double z)
{
#ifdef __SSE3__
	v = _mm_set_pd(y, x);
	val.z = z;
#else
	this->x = x;
	this->y = y;
	this->z = z;
#endif
}



// Operations that treat this point as a vector from (0, 0, 0):
double Point::Dot(const Point &point) const
{
#ifdef __SSE3__
	__m128d b = v * point.v;
	b = _mm_hadd_pd(b, b);
	return reinterpret_cast<double &>(b) + val.z * point.val.z;
#else
	return x * point.x + y * point.y + z * point.z;
#endif
}



Point Point::Cross(const Point &point) const
{
#ifdef __SSE3__
	// For 3D cross product
	return Point(
		val.y * point.val.z - val.z * point.val.y,
		val.z * point.val.x - val.x * point.val.z,
		val.x * point.val.y - val.y * point.val.x);
#else
	return Point(
		y * point.z - z * point.y,
		z * point.x - x * point.z,
		x * point.y - y * point.x);
#endif
}



double Point::Length() const
{
#ifdef __SSE3__
	__m128d b = v * v;
	b = _mm_hadd_pd(b, b);
	double result = reinterpret_cast<double &>(b) + val.z * val.z;
	return sqrt(result);
#else
	return sqrt(x * x + y * y + z * z);
#endif
}



double Point::LengthSquared() const
{
	return Dot(*this);
}



Point Point::Unit() const
{
#ifdef __SSE3__
	double lengthSquared = LengthSquared();
	if(!lengthSquared)
		return Point(1., 0., 0.);
	double scale = 1. / sqrt(lengthSquared);
	return *this * scale;
#else
	double b = x * x + y * y + z * z;
	if(!b)
		return Point(1., 0., 0.);
	b = 1. / sqrt(b);
	return Point(x * b, y * b, z * b);
#endif
}



double Point::Distance(const Point &point) const
{
	return (*this - point).Length();
}



double Point::DistanceSquared(const Point &point) const
{
	return (*this - point).LengthSquared();
}



Point Point::Lerp(const Point &to, const double c) const
{
	return *this + (to - *this) * c;
}



// Absolute value of all coordinates.
Point abs(const Point &p)
{
#ifdef __SSE3__
	// Absolute value for doubles just involves clearing the sign bit.
	static const __m128d sign_mask = _mm_set1_pd(-0.);
	Point result(_mm_andnot_pd(sign_mask, p.v));
	result.val.z = fabs(p.val.z);
	return result;
#else
	return Point(abs(p.x), abs(p.y), abs(p.z));
#endif
}



// Take the min of the x, y, and z coordinates.
Point min(const Point &p, const Point &q)
{
#ifdef __SSE3__
	Point result(_mm_min_pd(p.v, q.v));
	result.val.z = min(p.val.z, q.val.z);
	return result;
#else
	return Point(min(p.x, q.x), min(p.y, q.y), min(p.z, q.z));
#endif
}



// Take the max of the x, y, and z coordinates.
Point max(const Point &p, const Point &q)
{
#ifdef __SSE3__
	Point result(_mm_max_pd(p.v, q.v));
	result.val.z = max(p.val.z, q.val.z);
	return result;
#else
	return Point(max(p.x, q.x), max(p.y, q.y), max(p.z, q.z));
#endif
}



#ifdef __SSE3__
// Private constructor, using a vector.
inline Point::Point(const __m128d &v)
	: v(v)
{
	val.z = 0.;
}
#endif
