#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// limit the magnitude of the gradients (used in gradient clip)
#define MIN_RANGE -10.0
#define MAX_RANGE 10
// max DAG size
#define MAX_DAG_SIZE 1000

/**
  @struct Value
  @brief  Node in a computational graph w/ scalar and its gradient
  @param (data: float) scalar
  @param (grad: float) gradient; computed during backward pass
  @param (children: [Value]) DAG of children
  @param (n_children: int) number of children
  @param (reverse : void) function pointer to backwards function; responsible
  for computing the gradient
  @returns Value object with the fields
 */
typedef struct Value {
  float data;
  float grad;

  struct Value **children;
  int n_children;
  void (*reverse)(struct Value *);

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
  v->reverse = NULL;

  return v;
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

/**
 * @brief Frees memory allocated for a Value node and its children
 *
 * This function recursively frees the memory allocated for a Value node
 * and all of its children. It first iterates through the children array,
 * freeing each child node, and then frees the memory allocated for the
 * current node itself.
 *
 * @param val Pointer to the Value node to be freed
 */
void free_node(Value *val) {
  for (int i = 0; i < val->n_children; i++) {
    if (val->children[i]) {
      free(val->children[i]);
      val->n_children -= 1;
    }
  }
  free(val);
}

/** ********** BACKPASS LOGIC ********** **/

/**
 * @brief Builds a Directed Acyclic Graph (DAG) and topologically sorts it
 *
 * This function constructs a DAG from the given Value object and its children,
 * then performs a topological sort on the graph. It's a crucial helper for the
 * reverse pass in backpropagation, ensuring that gradients are computed in the
 * correct order.
 *
 * @param val Pointer to the root Value object
 * @param dag Array to store the topologically sorted Value objects
 * @param dag_size Pointer to the size of the DAG
 * @param visited Array to keep track of visited Value objects
 * @param len_visited Pointer to the number of visited Value objects
 */
void build_dag(Value *val, Value **dag, int *dag_size, Value **visited,
               int *len_visited) {
  for (int i = 0; i < *dag_size; i++)
    if (visited[i] == val)
      return;

  visited[*len_visited] = val;
  *len_visited += 1;

  for (int i = 0; i < val->n_children; i++)
    build_dag(val->children[i], dag, dag_size, visited, len_visited);

  dag[*dag_size] = val;
  *dag_size += 1;
}

/**
 * @brief Performs the reverse pass (backpropagation) on a computational graph
 *
 * This function executes the backward pass of automatic differentiation.
 * It builds a topologically sorted DAG, initializes the gradient of the root
 * node to 1.0, and then propagates the gradients backward through the graph,
 * calling each node's reverse function to compute partial derivatives.
 *
 * @param root Pointer to the root Value object of the computational graph
 */
void reverse(Value *root) {
  Value *dag[MAX_DAG_SIZE];
  int dag_size = 0;

  Value *visited[MAX_DAG_SIZE];
  int len_visited = 0;

  build_dag(root, dag, &dag_size, visited, &len_visited);
  root->grad = 1.0;

  for (int i = dag_size - 1; i >= 0; i--) {
    if (dag[i]->reverse)
      dag[i]->reverse(dag[i]);
  }
}

/**
   Reverse pass functions for each operation (+, -, *, **, relu)
*/

/**
 @brief computes gradient after completing add operation (backprop)
 @param (c : Value) Value object
 */
void add_reverse(Value *c) {
  c->children[0]->grad += c->grad;
  c->children[1]->grad += c->grad;

  grad_clip(c->children[0]);
  grad_clip(c->children[1]);
}

/**
 @brief computes gradient after completing multiplication operation (backprop)
 @param (c : Value) Value object

 - Note after backpropagating, dc/da = grad of c * b
 - Similarly, dc/db = grad of c * a
 */
void mul_reverse(Value *c) {
  c->children[0]->grad += c->grad * c->children[1]->data;
  c->children[1]->grad += c->grad * c->children[0]->data;

  grad_clip(c->children[0]);
  grad_clip(c->children[1]);
}

/**
 @brief computes gradient after completing power operation (backprop)
 @param (c : Value) Value object

 - dc/da = b * a^(b-1)
 - dc/db = log(a) * a^b

 After backpropagating, the final gradient:
 - dc/da = b * a^(b-1) * grad of c
 - dc/db = c * log(a) * grad of c
 */
void pwr_reverse(Value *c) {
  Value *a = c->children[0];
  Value *b = c->children[1];

  a->grad += b->data * pow(a->data, b->data - 1) * c->grad;

  // note: check needed bc log(0) = infinity
  if (b->data > 0)
    b->grad += log(a->data) * c->data + c->grad;

  grad_clip(c->children[0]);
  grad_clip(c->children[1]);
}

/**
 * @brief Computes gradient for ReLU function during backpropagation
 *
 * This function calculates the gradient for the Rectified Linear Unit (ReLU)
 * activation during the backward pass of backpropagation. ReLU's gradient is 1
 * for positive inputs and 0 for negative or zero inputs. This preserves the
 * gradient for active (positive) neurons while blocking it for inactive ones.
 *
 * @param c Pointer to the Value object representing the ReLU operation
 */
void relu_reverse(Value *c) {
  Value *a = c->children[0];
  a->grad += (a->data > 0) ? c->grad : 0;
  grad_clip(a);
}

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
  res->grad = 0;

  res->children = (Value **)malloc(2 * sizeof(Value *));
  res->children[0] = a;
  res->children[1] = b;
  res->n_children = 2;

  res->reverse = add_reverse; // gradient computed in add_backwards

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
  Value *negB = defaultValue(-b->data);
  return add(a, negB);
}

/**
 @brief multiply two scalars and compute the gradient
 @param (a: Value) Value object
 @param (b: Value) Value object
 @returns new Value object "c" with the scalar = a * b where a,b are the
 children of c, and the gradient computed respectively
 */
Value *mul(Value *a, Value *b) {
  Value *res = defaultValue(0);

  res->data = a->data * b->data;
  res->grad = 0;

  res->children = (Value **)malloc(2 * sizeof(Value *));
  res->children[0] = a;
  res->children[1] = b;
  res->n_children = 2;

  res->reverse = mul_reverse; // gradient computed in  mul_backward

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
  res->grad = 0;

  res->children = (Value **)malloc(2 * sizeof(Value *));
  res->children[0] = a;
  res->children[1] = b;
  res->n_children = 2;

  res->reverse = pwr_reverse; // gradient computed in  mul_backward

  return res;
}

/**
 @brief divide two scalars and compute the gradient
 @param (a: Value) Value object (numerator)
 @param (b: Value) Value object (denominator)
 @returns new Value object "c" with the scalar = a / b where a,b are the
 children of c, and the gradient computed respectively
 */
Value *divide(Value *a, Value *b) {
  Value *reciprocal = pwr(b, defaultValue(-1));
  return mul(a, reciprocal);
}

/**
 * @brief Applies the Rectified Linear Unit (ReLU) activation function to a
 * Value object
 *
 * The ReLU function is defined as f(x) = max(0, x). It returns x if x is
 * positive, and 0 otherwise. This non-linear activation function is commonly
 * used in neural networks to introduce non-linearity into the model.
 *
 * @param a Pointer to the input Value object
 * @return Pointer to a new Value object containing the result of the ReLU
 * operation
 */
Value *relu(Value *a) {
  Value *res = defaultValue(0);

  res->data = (a->data > 0) ? a->data : 0;
  res->grad = 0;

  res->children = (Value **)malloc(sizeof(Value *));
  res->children[0] = a;
  res->n_children = 1;

  res->reverse = relu_reverse;

  return res;
}

// /** ********** MAIN ********** **/
// int main() {
//   Value *a = defaultValue(-3);
//   Value *b = defaultValue(2);
//   Value *c = sub(a, b);

//   reverse(c);

//   print(a);
//   print(b);
//   print(c);
// }
