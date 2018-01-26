// Copyright 2018 Jim Turner
//
// Licensed under the [Apache License, Version 2.0](LICENSE-APACHE) or the [MIT
// license](LICENSE-MIT), at your option. You may not use this file except in
// compliance with those terms.

//! The `ndarray_expr` crate provides types to create and evaluate expression
//! trees of operations on `ndarray` arrays.
//!
//! ## Differences from normal evaluation
//!
//! - Evaluation of the expression tree does not create any intermediate arrays.
//!
//! - Cobroadcasting of array shapes is supported and follows [NumPy's
//!   broadcasting behavior][].
//!
//! - Currently, to broadcast arrays with different numbers of dimensions, you
//!   must convert them to [`IxDyn`][] dimensionality or use
//!   [`.insert_axis()`][] to convert them to the same dimensionality. This
//!   limitation may be removed in the future.
//!
//! - Currently, arrays are required to have the same element type. It
//!   shouldn't be too hard to remove this limitation.
//!
//! - Currently, the element type needs to implement [`Copy`], primarily so
//!   that [`.eval()`][] can call [`Array::uninitialized()`][]. This limitation
//!   isn't fundamental to the expression tree idea, though, and could be
//!   eliminated.
//!
//! - Currently, [`.eval()`][] always has to allocate a new result array
//!   because all of the inputs are [`ArrayView`][]s. It might be possible to
//!   change the implementation to allow reuse of arguments that are
//!   [`Array`][]s, but that requires more investigation. If you have already
//!   allocated a result array, you can use [`.eval_assign()`][] to avoid any
//!   more allocations.
//!
//! [NumPy's broadcasting behavior]: https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html#general-broadcasting-rules
//! [`IxDyn`]: https://docs.rs/ndarray/0.11/ndarray/type.IxDyn.html
//! [`.insert_axis()`]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.insert_axis
//! [`Copy`]: https://doc.rust-lang.org/stable/std/marker/trait.Copy.html
//! [`Array::uninitialized()`]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.uninitialized
//! [`.eval()`]: trait.Expression.html#method.eval
//! [`Array`]: https://docs.rs/ndarray/0.11/ndarray/type.Array.html
//! [`ArrayView`]: https://docs.rs/ndarray/0.11/ndarray/type.ArrayView.html
//! [`.eval_assign()`]: trait.Expression.html#method.eval_assign
//!
//! ## Performance
//!
//! I haven't done any benchmarking with this crate yet, although I did write a
//! very similar expression tree for `Vec<T>`/`&[T]` and in release mode the
//! compiler was able to completely evaluate expressions with constant inputs
//! *at compile time*. This suggests to me that the compiler is good at
//! inlining expression trees like this to generate fast code.
//!
//! ## Examples
//!
//! This is an example illustrating how to create and evaluate a simple
//! expression:
//!
//! ```
//! #[macro_use(array)]
//! extern crate ndarray;
//! extern crate ndarray_expr;
//!
//! use ndarray_expr::{Expression, ExpressionExt};
//!
//! # fn main() {
//! let a = array![1, 2, 3];
//! let b = array![4, -5, 6];
//! let c = array![-3, 7, -5];
//! assert_eq!(
//!     (-a.as_expr() * b.as_expr() + c.expr_mapv(i32::abs) * a.as_expr()).eval(),
//!     array![-1, 24, -3],
//! );
//! # }
//! ```
//!
//! This example illustrates cobroadcasting of two 2D arrays:
//!
//! ```
//! # #[macro_use(array)]
//! # extern crate ndarray;
//! # extern crate ndarray_expr;
//! # use ndarray_expr::{Expression, ExpressionExt};
//! # fn main() {
//! let a = array![
//!     [1],
//!     [2],
//!     [3]
//! ];
//! let b = array![
//!     [4, -5, 6]
//! ];
//! assert_eq!(
//!     (a.as_expr() + b.as_expr()).eval(),
//!     array![
//!         [5, -4, 7],
//!         [6, -3, 8],
//!         [7, -2, 9]
//!     ],
//! );
//! # }
//! ```
//!
//! This example illustrates cobroadcasting of a 3D array and a 2D array:
//!
//! ```
//! # #[macro_use(array)]
//! # extern crate ndarray;
//! # extern crate ndarray_expr;
//! # use ndarray_expr::{Expression, ExpressionExt};
//! # fn main() {
//! let a = array![[[1], [2], [3]], [[-7], [3], [-9]]].into_dyn();
//! let b = array![[4, -5, 6]].into_dyn();
//! assert_eq!(
//!     (a.as_expr() - b.as_expr()).eval(),
//!     array![
//!         [[-3, 6, -5], [-2, 7, -4], [-1, 8, -3]],
//!         [[-11, -2, -13], [-1, 8, -3], [-13, -4, -15]]
//!     ].into_dyn(),
//! );
//! # }
//! ```

extern crate ndarray;

use ndarray::prelude::*;
use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Not, Rem, Sub};

/// An expression of arrays.
pub trait Expression: Sized {
    /// Type of elements in the arrays.
    type Elem: Copy;

    /// Dimension of the result of the expression.
    type Dim: Dimension;

    /// Iterator over the elements in the arrays in the expression.
    type Iter: Iterator;

    /// Returns an iterator over the elements in the arrays in the expression.
    fn iter(&self) -> Self::Iter;

    /// Returns the number of dimensions (axes) of the result.
    fn ndim(&self) -> usize;

    /// Returns a clone of the raw dimension.
    fn raw_dim(&self) -> Self::Dim;

    /// Returns the shape of the result.
    fn shape(&self) -> &[usize];

    /// Broadcast into a larger shape, if possible.
    fn broadcast_move(self, shape: Self::Dim) -> Option<Self>;

    /// Applies the expression to individual elements of the arrays.
    fn eval_elems(&self, elems: <Self::Iter as Iterator>::Item) -> Self::Elem;

    /// Applies the expression to the arrays, returning the result.
    ///
    /// This method does not allocate any intermediate arrays; it allocates
    /// only the single output array.
    fn eval(&self) -> Array<Self::Elem, Self::Dim> {
        let mut out = unsafe { Array::uninitialized(self.raw_dim()) };
        self.eval_assign(out.view_mut());
        out
    }

    /// Applies the expression to the arrays, assigning the result to `out`.
    ///
    /// This method does not allocate any memory.
    fn eval_assign<'a>(&self, mut out: ArrayViewMut<'a, Self::Elem, Self::Dim>) {
        out.iter_mut()
            .zip(self.iter())
            .for_each(|(out, elems)| *out = self.eval_elems(elems));
    }

    /// Creates an expression that calls `f` by value on each element.
    fn mapv_into<F>(self, f: F) -> UnaryOpExpr<F, Self>
    where
        F: Fn(Self::Elem) -> Self::Elem,
    {
        UnaryOpExpr::new(f, self)
    }
}

/// Convenience extension methods for `ArrayBase`.
pub trait ExpressionExt<A, D>
where
    D: Dimension,
{
    /// Creates an expression view of `self`.
    fn as_expr(&self) -> ArrayViewExpr<A, D>;

    /// Creates an expression that calls `f` by value on each element.
    fn expr_mapv<F>(&self, f: F) -> UnaryOpExpr<F, ArrayViewExpr<A, D>>
    where
        F: Fn(A) -> A,
        A: Copy;
}

impl<A, S, D> ExpressionExt<A, D> for ArrayBase<S, D>
where
    S: ndarray::Data<Elem = A>,
    D: Dimension,
{
    fn as_expr(&self) -> ArrayViewExpr<A, D> {
        ArrayViewExpr::new(self.view())
    }

    fn expr_mapv<F>(&self, f: F) -> UnaryOpExpr<F, ArrayViewExpr<A, D>>
    where
        F: Fn(A) -> A,
        A: Copy,
    {
        self.as_expr().mapv_into(f)
    }
}

/// An expression wrapper for an `ArrayView`.
#[derive(Clone, Debug)]
pub struct ArrayViewExpr<'a, A: 'a, D: 'a>(ArrayView<'a, A, D>)
where
    D: Dimension;

impl<'a, A, D> ArrayViewExpr<'a, A, D>
where
    D: Dimension,
{
    /// Creates a new expression from the view.
    pub fn new(view: ArrayView<'a, A, D>) -> Self {
        ArrayViewExpr(view)
    }
}

impl<'a, A, D> Expression for ArrayViewExpr<'a, A, D>
where
    A: Copy,
    D: Dimension,
{
    type Elem = A;
    type Dim = D;
    type Iter = ndarray::iter::Iter<'a, A, D>;

    fn iter(&self) -> Self::Iter {
        self.0.clone().into_iter()
    }

    fn ndim(&self) -> usize {
        self.0.ndim()
    }

    fn raw_dim(&self) -> D {
        self.0.raw_dim()
    }

    fn shape(&self) -> &[usize] {
        self.0.shape()
    }

    fn broadcast_move(self, shape: D) -> Option<Self> {
        self.0.broadcast(shape.clone()).map(|bc| {
            // Copy strides from broadcasted view.
            let mut strides = D::zero_index_with_ndim(shape.ndim());
            strides
                .slice_mut()
                .iter_mut()
                .zip(bc.strides())
                .for_each(|(out_s, bc_s)| *out_s = *bc_s as usize);
            // Create a new `ArrayView` with the shape and strides. This is the
            // only way to keep the same lifetime since there is no
            // `broadcast_move` method for `ArrayView`.
            ArrayViewExpr(unsafe { ArrayView::from_shape_ptr(shape.strides(strides), bc.as_ptr()) })
        })
    }

    #[inline]
    fn eval_elems(&self, elems: <Self::Iter as Iterator>::Item) -> A {
        *elems
    }
}

/// An expression with a single argument.
#[derive(Clone, Debug)]
pub struct UnaryOpExpr<F, E> {
    oper: F,
    inner: E,
}

impl<F, E> UnaryOpExpr<F, E>
where
    F: Fn(E::Elem) -> E::Elem,
    E: Expression,
{
    /// Returns a new expression applying `oper` to `inner`.
    pub fn new(oper: F, inner: E) -> Self {
        UnaryOpExpr { oper, inner }
    }
}

impl<F, E> Expression for UnaryOpExpr<F, E>
where
    F: Fn(E::Elem) -> E::Elem,
    E: Expression,
{
    type Elem = E::Elem;
    type Dim = E::Dim;
    type Iter = E::Iter;

    fn iter(&self) -> Self::Iter {
        self.inner.iter()
    }

    fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    fn raw_dim(&self) -> Self::Dim {
        self.inner.raw_dim()
    }

    fn shape(&self) -> &[usize] {
        self.inner.shape()
    }

    fn broadcast_move(self, shape: Self::Dim) -> Option<Self> {
        let UnaryOpExpr { oper, inner, .. } = self;
        inner
            .broadcast_move(shape)
            .map(|new_inner| UnaryOpExpr::new(oper, new_inner))
    }

    #[inline]
    fn eval_elems(&self, elems: <Self::Iter as Iterator>::Item) -> Self::Elem {
        (self.oper)(self.inner.eval_elems(elems))
    }
}

/// An expression with two arguments.
#[derive(Clone, Debug)]
pub struct BinaryOpExpr<F, E1, E2> {
    oper: F,
    left: E1,
    right: E2,
}

/// Broadcast the shapes together, follwing the behavior of NumPy.
fn broadcast<D: Dimension>(shape1: &[usize], shape2: &[usize]) -> Option<D> {
    // Zip the dims in reverse order, adding `&1`s to the shorter one until
    // they're the same length.
    let zipped = if shape1.len() < shape2.len() {
        shape1
            .iter()
            .rev()
            .chain(std::iter::repeat(&1))
            .zip(shape2.iter().rev())
    } else {
        shape2
            .iter()
            .rev()
            .chain(std::iter::repeat(&1))
            .zip(shape1.iter().rev())
    };
    let mut out = D::zero_index_with_ndim(std::cmp::max(shape1.len(), shape2.len()));
    for ((&len1, &len2), len_out) in zipped.zip(out.slice_mut().iter_mut().rev()) {
        if len1 == len2 {
            *len_out = len1;
        } else if len1 == 1 {
            *len_out = len2;
        } else if len2 == 1 {
            *len_out = len1;
        } else {
            return None;
        }
    }
    Some(out)
}

impl<F, E1, E2> BinaryOpExpr<F, E1, E2>
where
    F: Fn(E1::Elem, E2::Elem) -> E1::Elem,
    E1: Expression,
    E2: Expression<Elem = E1::Elem, Dim = E1::Dim>,
{
    /// Returns a new expression applying `oper` to `left` and `right`.
    ///
    /// Returns `None` if the shapes of the arrays cannot be broadcast
    /// together. Note that the broadcasting is more general than
    /// `ArrayBase.broadcast()`; cobroadcasting is supported.
    pub fn new(oper: F, left: E1, right: E2) -> Option<Self> {
        broadcast::<E1::Dim>(left.shape(), right.shape()).map(|shape| {
            let left = left.broadcast_move(shape.clone()).unwrap();
            let right = right.broadcast_move(shape.clone()).unwrap();
            BinaryOpExpr { oper, left, right }
        })
    }
}

impl<F, E1, E2> Expression for BinaryOpExpr<F, E1, E2>
where
    F: Fn(E1::Elem, E2::Elem) -> E1::Elem,
    E1: Expression,
    E2: Expression<Elem = E1::Elem, Dim = E1::Dim>,
{
    type Elem = E1::Elem;
    type Dim = E1::Dim;
    type Iter = std::iter::Zip<E1::Iter, E2::Iter>;

    fn iter(&self) -> Self::Iter {
        self.left.iter().zip(self.right.iter())
    }

    fn ndim(&self) -> usize {
        self.left.ndim()
    }

    fn raw_dim(&self) -> Self::Dim {
        self.left.raw_dim()
    }

    fn shape(&self) -> &[usize] {
        self.left.shape()
    }

    fn broadcast_move(self, shape: Self::Dim) -> Option<Self> {
        let BinaryOpExpr {
            oper, left, right, ..
        } = self;
        match (
            left.broadcast_move(shape.clone()),
            right.broadcast_move(shape),
        ) {
            (Some(new_left), Some(new_right)) => BinaryOpExpr::new(oper, new_left, new_right),
            _ => None,
        }
    }

    #[inline]
    fn eval_elems(&self, elems: <Self::Iter as Iterator>::Item) -> Self::Elem {
        (self.oper)(
            self.left.eval_elems(elems.0),
            self.right.eval_elems(elems.1),
        )
    }
}

macro_rules! impl_unary_op {
    ($trait:ident, $method:ident, $($header:tt)*) => {
        $($header)*
        where
            Self: Expression,
            <Self as Expression>::Elem: $trait<Output = <Self as Expression>::Elem>,
        {
            type Output = UnaryOpExpr<
                fn(<Self as Expression>::Elem) -> <Self as Expression>::Elem,
                Self,
            >;

            fn $method(self) -> Self::Output {
                UnaryOpExpr::new($trait::$method, self)
            }
        }
    }
}

macro_rules! impl_unary_op_all {
    ($trait:ident, $method:ident) => {
        impl_unary_op!(
            $trait, $method,
            impl<'a, A, D: Dimension> $trait for ArrayViewExpr<'a, A, D>
        );
        impl_unary_op!(
            $trait, $method,
            impl<F, E> $trait for UnaryOpExpr<F, E>
        );
        impl_unary_op!(
            $trait, $method,
            impl<F, E1, E2> $trait for BinaryOpExpr<F, E1, E2>
        );
    }
}

impl_unary_op_all!(Neg, neg);
impl_unary_op_all!(Not, not);

macro_rules! impl_binary_op {
    ($trait:ident, $method:ident, $($header:tt)*) => {
        $($header)*
        where
            Self: Expression,
            <Self as Expression>::Elem: $trait<Output = <Self as Expression>::Elem>,
            Rhs: Expression<Elem = <Self as Expression>::Elem, Dim = <Self as Expression>::Dim>,
        {
            type Output = BinaryOpExpr<
                fn(<Self as Expression>::Elem, <Self as Expression>::Elem)
                   -> <Self as Expression>::Elem,
                Self,
                Rhs,
            >;

            fn $method(self, rhs: Rhs) -> Self::Output {
                // Extra type annotation is necessary to prevent compile error
                // due to incorrect inference.
                BinaryOpExpr::<fn(_, _) -> _, _, _>::new($trait::$method, self, rhs).unwrap()
            }
        }
    }
}

macro_rules! impl_binary_op_all {
    ($trait:ident, $method:ident) => {
        impl_binary_op!(
            $trait, $method,
            impl<'a, A, D: Dimension, Rhs> $trait<Rhs> for ArrayViewExpr<'a, A, D>
        );
        impl_binary_op!(
            $trait, $method,
            impl<F, E, Rhs> $trait<Rhs> for UnaryOpExpr<F, E>
        );
        impl_binary_op!(
            $trait, $method,
            impl<F, E1, E2, Rhs> $trait<Rhs> for BinaryOpExpr<F, E1, E2>
        );
    }
}

impl_binary_op_all!(Add, add);
impl_binary_op_all!(BitAnd, bitand);
impl_binary_op_all!(BitOr, bitor);
impl_binary_op_all!(BitXor, bitxor);
impl_binary_op_all!(Div, div);
impl_binary_op_all!(Mul, mul);
impl_binary_op_all!(Rem, rem);
impl_binary_op_all!(Sub, sub);
