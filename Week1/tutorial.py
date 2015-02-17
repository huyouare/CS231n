def quicksort(arr):
  if len(arr) <= 1:
    return arr
  pivot = arr[len(arr) / 2]
  left = [x for x in arr if x < pivot]
  middle = [x for x in arr if x == pivot]
  right = [x for x in arr if x > pivot]
  return quicksort(left) + middle + quicksort(right)

print quicksort([3,6,8,10,1,2,1])


## Basic types
x = 3
print type(3)
print x
print x + 1
print x - 1
print x * 2
print x ** 2
# Caret operator does an XOR operation
x += 1
print x
x *= 2
print x
y = 2.5
print type(y)
print y, y + 1, y * 2, y ** 2

t = True
f = False
print type(t)
print t and f
print t or f
print not t
print t != f # Logical XOR
print t ^ f # Also XOR

hello = 'hello'
world = "world"
print hello
print len(hello)
hw = hello + ' ' + world
print hw
hw12 = '%s %s %d' % (hello, world, 12)
print hw12

s = "hello"
print s.capitalize()
print s.upper()
print s.rjust(7) # Right-justify
print s.center(7) # Center a string, padding with spaces
print s.replace('1', '(ell)')
print '  world  '.strip()


## Lists
xs = [3, 1, 2]
print xs, xs[2]
print xs[-1]
xs[2] = 'foo'
print xs
xs.append('bar')
print xs
x = xs.pop() # Remove and return last element
print x, xs

nums = range(5)
print nums
print nums[2:4] # End index exclusive
print nums[2:]
print nums[:2]
print nums[:]
print nums[:-1]
nums[2:4] = [8, 9]
print nums

# Loops
animals = ['cat', 'dog', 'monkey']
for animal in animals:
  print animal

for idx, animal in enumerate(animals):
  print '#%d: %s' % (idx + 1, animal)

# List Comprehensions
nums = [0, 1, 2, 3, 4]
squares = []
for x in nums:
  squares.append(x ** 2)
print squares

nums = [0, 1, 2, 3, 4]
squares = [x ** 2 for x in nums]
print squares
even_squares = [x ** 2 for x in nums if x % 2 == 0]
print even_squares


## Dictionaries
d = {'cat': 'cute', 'dog': 'furry'}
print d['cat']
print 'cat' in d # Check if dictionary has key => True
d['fish'] = 'wet'
print d['fish']
# print d['monkey']
print d.get('monkey', 'N/A') # Gets key with default value
print d.get('fish', 'N/A') 
del d['fish']
print d.get('fish', 'N/A')

# Loops
d = {'person': 2, 'cat': 4, 'spider': 8}
for animal in d:
  legs = d[animal]
  print 'A %s has %d legs' % (animal, legs)

for animal, legs in d.iteritems():
  print 'A %s has %d legs' % (animal, legs)

# List Comprehension to Dictionary
even_num_to_square = {x: x ** 2 for x in nums if x % 2 == 0}
print even_num_to_square

## Sets
animals = {'cat', 'dog'}
print 'cat' in animals
print 'fish' in animals
animals.add('fish')
print 'fish' in animals
print len(animals)
animals.add('cat')
print len(animals)
animals.remove('cat')
print len(animals)

for idx, animal in enumerate(animals):
  print '#%d: %s' % (idx + 1, animal)

from math import sqrt
nums = {int(sqrt(x)) for x in range(30)}
print nums

## Tuples
# immutable ordered list of values, 
# can be used as keys in dictionary or elements of sets
d = {(x, x + 1): x for x in range(10)}
t = (5, 6)
print type(t)
print d[t]
print d[(1, 2)]

## Functions
def sign(x):
  if x > 0:
    return 'positive'
  elif x < 0:
    return 'negative'
  else:
    return 'zero'

for x in [-1, 0, 1]:
  print sign(x)

def hello(name, loud=False):
  if loud:
    print 'HELLO, %s' % name.upper()
  else:
    print 'Hello, %s!' % name
hello('Bob')
hello('Fred', loud=True)


## Classes
class Greeter:

  # Constructor
  def __init__(self, name):
    self.name = name # instance variable

  # Instance method
  def greet(self, loud=False):
    if loud:
      print 'HELLO, %s!' % self.name.upper()
    else:
      print 'Hello, %s' % self.name

g = Greeter('Fred')
g.greet()
g.greet(loud=True)
