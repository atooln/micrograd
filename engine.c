#include <math.h>
#include <stdio.h>
#include <stdlib.h>

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
  print(a);
  print(b);
}