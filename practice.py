import theano
import theano.tensor as T 
k = T . iscalar('k')
A = T . vector( 'A')
outputs, updates = theano.scan(lambda result, A : result * A,
             non_sequences = A, 
             outputs_info=T.ones_like(A), n_steps = k)
print(outputs)
result = outputs
#print(result)
fn_Ak = theano.function([A,k],result, updates=updates )
print (fn_Ak([0,1,3,4,5,6,7], 2 ))
print(range(3))
#print(outputs)
#print(updates)