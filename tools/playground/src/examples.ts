/**
 * Curated example programs for the playground.
 */

export interface Example {
  name: string;
  code: string;
}

export const EXAMPLES: Example[] = [
  {
    name: "Hello World",
    code: `// Hello World in Adam
println("Hello, World!")
`,
  },
  {
    name: "Fibonacci",
    code: `// Recursive Fibonacci
fn fibonacci(n) {
    if n <= 1 {
        n
    } else {
        fibonacci(n - 1) + fibonacci(n - 2)
    }
}

println(fibonacci(20))
`,
  },
  {
    name: "Pipe Operator",
    code: `// Pipe operator chains transformations left-to-right
fn double(x) { x * 2 }
fn add_one(x) { x + 1 }
fn square(x) { x * x }

let result = 5 |> double |> add_one |> square
println(result)
`,
  },
  {
    name: "Closures",
    code: `// Closures capture their enclosing environment
fn make_counter(start) {
    let count = start
    |step| {
        count = count + step
        count
    }
}

let counter = make_counter(0)
println(counter(1))
println(counter(1))
println(counter(5))
`,
  },
  {
    name: "Arrays",
    code: `// Array operations
let numbers = [10, 20, 30, 40, 50]

println(len(numbers))
println(numbers[0])
println(numbers[4])

// Sum with a loop
let total = 0
for x in numbers {
    total = total + x
}
println(total)
`,
  },
  {
    name: "Calculator",
    code: `// Calculator — closures, pipes, arrays, strings, booleans

fn double(x) { x * 2 }
fn add_one(x) { x + 1 }

// Pipe operator
let result = 5 |> double |> add_one
println(result)

// Array operations
let arr = [10, 20, 30, 40, 50]
println(len(arr))
println(arr[2])

// Closures
fn make_adder(n) {
    |x| n + x
}
let add5 = make_adder(5)
println(add5(10))

// String concatenation
let greeting = "Hello" + " " + "World!"
println(greeting)

// Boolean logic
let x = true && false
let y = true || false
println(x)
println(y)
`,
  },
];
