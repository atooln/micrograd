#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/**
 @brief
 @param
 @returns
 */

// Range for gradient clip
#define MIN_RANGE -10.0
#define MAX_RANGE 10

/**
  @struct Value
  @brief  Node in a computational graph w/ scalar and its gradient

  @param (data: float) scalar
  @param (grad: float) gradient; computed during backward pass
  @param (children: [Value]) DAG of children
  @param (n_children: int) number of children
  @param (backward : void) function pointer to backwards function; responsible
  for computing the gradient

  @returns Value object with the fields
 */
typedef struct Value {
  float data;
  float grad;

  struct Value **children;
  int n_children;
  void (*backward)(struct Value *);

} Value;

/**
  @brief initialize Value object by floating point number
  @param (x) float to be converted into a Value object
  @returns Value object
 */
Value *defaultValue(float x) {
  Value *v = (Value *)malloc(sizeof(Value));

  v->data = x;
  v->grad = 0;
  v->children = NULL;
  v->n_children = 0;
  v->backward = NULL;

  return v;
}

/**
 @brief build DAG and topologically sort it (helper for backwards pass)
 @param
 @returns
*/

/**
  @brief backwards pass function
  @param
  @returns
 */

/** ********** BACKPASSES FOR OPERATORS ********** **/

/**
   TODO: create backwards pass functions for each operation (+, -, *, **, relu)
*/

/** ********** OPERATORS ********** **/

/**
  TODO: build operations (+, -, *, **, relu)
 */

/**
 @brief add two scalars and compute the gradient
 @param (a: Value) Value object
 @param (b: Value) Value object
 @returns new Value object "c" with the scalar = a + b where a,b are the
 children of c, and the gradient computed respectively
 */
Value *add(Value *a, Value *b) {
  Value *res = defaultValue(0);

  res->data = a->data + b->data;
  // res->grad = 0;
  res->children = (Value **)malloc(2 * sizeof(Value *));

  res->children[0] = a;
  res->children[1] = b;
  res->n_children = 2;
  // res->backward = add_backward; //gradient computed in add_backwards

  return res;
}

/**
 @brief subtract two scalars and compute the gradient
 @param (a: Value) Value object
 @param (b: Value) Value object
 @returns new Value object "c" with the scalar = a - b where a,b are the
 children of c, and the gradient computed respectively
 */
Value *sub(Value *a, Value *b) {
  Value *res = defaultValue(0);

  res->data = a->data - b->data;
  // res->grad = 0;
  res->children = (Value **)malloc(2 * sizeof(Value *));

  res->children[0] = a;
  res->children[1] = b;
  res->n_children = 2;
  // res->backward = sub_backward; //gradient computed in sub_backward

  return res;
}

/**
 @brief subtract two scalars and compute the gradient
 @param (a: Value) Value object
 @param (b: Value) Value object
 @returns new Value object "c" with the scalar = a * b where a,b are the
 children of c, and the gradient computed respectively
 */
Value *mul(Value *a, Value *b) {
  Value *res = defaultValue(0);

  res->data = a->data * b->data;
  // res->grad = 0;
  res->children = (Value **)malloc(2 * sizeof(Value *));

  res->children[0] = a;
  res->children[1] = b;
  res->n_children = 2;
  // res->backward = mul_backward; //gradient computed in  mul_backward

  return res;
}

/**
 @brief raise one scalar to the power of the other and compute the gradient
 @param (a: Value) Value object
 @param (b: Value) Value object
 @returns new Value object "c" with the scalar = a ^ b where a,b are the
 children of c, and the gradient computed respectively
 */
Value *pwr(Value *a, Value *b) {
  Value *res = defaultValue(0);

  res->data = powf(a->data, b->data);
  // res->grad = 0;
  res->children = (Value **)malloc(2 * sizeof(Value *));

  res->children[0] = a;
  res->children[1] = b;
  res->n_children = 2;
  // res->backward = mul_backward; //gradient computed in  mul_backward

  return res;
}

/** ********** UTILS ********** **/

/**
  @brief print value and gradient
  @param (obj: Value) Value object
 */
void print(Value *obj) {
  printf("Value: %f, Gradient: %f \n", obj->data, obj->grad);
}

/**
 @brief clips the gradient if it exceeds a certain range
 @param (obj: Value) Value object
*/
void grad_clip(Value *obj) {
  if (obj->grad < MIN_RANGE) {
    obj->grad = MIN_RANGE;
  } else if (obj->grad > MAX_RANGE) {
    obj->grad = MAX_RANGE;
  }
}

/** ********** MAIN ********** **/
int main() {
  Value *a = defaultValue(3);
  Value *b = defaultValue(2);
  Value *c = add(a, b);
  print(a);
  print(b);
  print(c);
}