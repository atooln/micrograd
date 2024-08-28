#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Object to apply numerical computations
typedef struct Value {
  float data; // data
  float grad; // computed gradient

  struct Value **children; // children of this Value object
  int n_children;          // children
  void (*backward)(
      struct Value *); // computes gradients // INTERESTING FUNCTION

} Value;

// Make default value object
Value *defaultValue(float x) {
  Value *v = (Value *)malloc(sizeof(Value));

  v->data = x;
  v->grad = 0;
  v->children = NULL;
  v->n_children = 0;
  v->backward = NULL;

  return v;
}

// MAIN
int main() {}