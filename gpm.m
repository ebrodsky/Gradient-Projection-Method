x = [.5;.5;.5;.5];
%x = [1;0;0;0];
k = 0;
del = 0.00001;
W = findW(x, del);
done = 0;
I = eye(4);
gradObj(x)
Ws = [W];
eta = 1.2;
epsilon = 0.4;
xs = [x];
obj_vals = [objective(x)];
grad_mags = [norm(gradObj(x),2)];
mat = [];
while(~done)
    %step 1
    fprintf("step 1")
    A = createA(W, x);
    q = length(W);
    %step 2
    fprintf("step 2")
    d = (I-A'*inv(A*A')*A)*(-gradObj(x)');
    %step 3
    fprintf("step 3")
    norm(d,2)
    s3_repeat = 0;
    if(norm(d,2) <= del) % if ||d|| > del, go to step 4
        w = inv(A*A')*A*(-gradObj(x)'); %w is the combined vector of mu and lambda
        u = w(1:q)
        if(all(u >= 0))
            break; %all of the u_i's are nonnegative, so break and terminate
        else
            s3_repeat = 1;
            [m,i] = min(u) %find the mimimum value, use value's index to remove the constraint at A(i) from W.
            if(isequal(A(i,:),g1_grad(x)))
                W = removeConstraint(W, 1); % if A(min val index) = constraint #1, remove the number 1 from W.
            elseif(isequal(A(i,:),g2_grad(x))) 
                W = removeConstraint(W, 2);
            elseif(isequal(A(i,:),g3_grad(x)))
                W = removeConstraint(W, 3);
            elseif(isequal(A(i,:),g4_grad(x)))
                W = removeConstraint(W, 4);
            end
        end
    end
    if(s3_repeat == 0) %if we're not repeating from step 1 after taking out a constraint in step 3
        %step 4
        fprintf("step 4")
        alpha = 0.5; %4a
        while 1
            while 1 %4b
                y = x + alpha*d;
                z_opt = line_opt(W, A, y); %find optimal z
                %fprintf("linobjective")
                lin_objective(W, y, A, z_opt);
                if(lin_objective(W, y, A, z_opt) < del)
                    xbar = y + A'*z_opt
                    break;
                else
                    %alpha is too large.
                    %fprintf("alpha too large 1");
                    alpha = alpha / eta;
                end

            end
            if(all(xbar >= 0) && objective(xbar) <= objective(x) + epsilon*alpha*gradObj(x)*d) %4c
                x = xbar;
                obj_vals = [obj_vals objective(x)];
                xs = [xs x];
                break;
            else
                %fprintf("alpha too large 2");
                alpha = alpha / eta;
            end
        end
        grad_mags = [grad_mags norm(gradObj(x), 2)];
        %step 5
        fprintf("step 5")
        W = findW(x, del);
        mat = addActivityColumn(W, mat);
        k = k + 1
    end

end

    %By some miracle, this actually converged.
    
    %Question 2a
    figure(1);
    plot([1:1:k+1], obj_vals)
    xlabel('Iteration')
    ylabel('Objective function value')
%     I chose my variables based on lecture slides and homework solutions for similar algorithms,
%     particularly for the backtracking line search. I used eta = 1.2 and epsilon = 0.4 as in the homework
%     solutions, and I used alpha_init as 0.5 because it seemed like a standard choice. For the backtracking
%     line search unconstrained optimization, I chose 0.005 because that also seemed like a standard choice
%     specifically for a constant step size.
    
    %Question 2b
    figure(2);
    image(256*mat)
%     The working set seemed to increase in cardinality as k also increased. The end result had 
%     3 out of the 5 constraints active, though one of them is the default equality constraint 
%     which is always active by definition. Similarly, the objective
%     function decreased as k increased, which it to be expected since this
%     indicates that the optimization algorithm works. The resulting x* was
%     x* = [0.1110;0.0000;0.9938;0.0000], and the minimum value was -4.5555.
%     At that final x*, we see that the 2nd and the 4th inequality
%     constraints were active.

    %Question 2c
        df = gradObj(x);
        dh = h1_grad(x);
        dg = [g1_grad(x); g2_grad(x); g3_grad(x); g4_grad(x)];
        A = dg;
        A = [A,dh'];
        y = -df;
        b = A\y'

function res = addActivityColumn(W, mat)
    vec = [0;0;0;0;0];
    for c = 1:length(W)
        if W(c) == 1
            vec = vec + [1;0;0;0;0];
        elseif W(c) == 2
            vec = vec + [0;1;0;0;0];
        elseif W(c) == 3
            vec = vec + [0;0;1;0;0];
        elseif W(c) == 4
            vec = vec + [0;0;0;1;0];
        end 
    end
    vec = vec + [0;0;0;0;1];
    res = [mat,vec];

end

function set = removeConstraint(W, m)
for c = 1:length(W)
    if W(c) == m
        W(c) = [];
        break;
    end
end
set = W;
end

function val = g1(x)
val = -x(1);
end
function val = g2(x)
val = -x(2);
end
function val = g3(x)
val = -x(3);
end
function val = g4(x)
val = -x(4);
end
function val = h1(x)
val = x(1)^2 + x(2)^2 + x(3)^2 + x(4)^2 -1;
end

function vec = g1_grad(x)
vec = [-1,0,0,0];
end
function vec = g2_grad(x)
vec = [0,-1,0,0];
end
function vec = g3_grad(x)
vec = [0,0,-1,0];
end
function vec = g4_grad(x)
vec = [0,0,0,-1];
end
function vec = h1_grad(x)
vec = [2*x(1), 2*x(2), 2*x(3), 2*x(4)];
end

function val = lin_objective(W, y, A, z)

val = 0;
v = y + A'*z;
for c = 1:length(W)
    if W(c) == 1
        val = val + g1(v)^2;
    elseif W(c) == 2
        val = val + g2(v)^2;
    elseif W(c) == 3
        val = val + g3(v)^2;
    elseif W(c) == 4
        val = val + g4(v)^2;
    end
end

val = val + h1(v)^2;

end

%calculate the gradient of the backtracking line search problem
function val = grad_lin_obj(W, y, A, z)

qm = size(A,1);
val = zeros(1,qm);
v = y + A'*z;
for c = 1:length(W)
    if W(c) == 1
        val = val + 2*g1(v)*g1_grad(v)*A';
    elseif W(c) == 2
        val = val + 2*g2(v)*g2_grad(v)*A';
    elseif W(c) == 3
        val = val + 2*g3(v)*g3_grad(v)*A';
    elseif W(c) == 4
        val = val + 2*g4(v)*g4_grad(v)*A';
    end
end

val = val + 2*h1(v)*h1_grad(v)*A';

end

% run gradient descent to determine projection back onto constraints.
function val = line_opt(W, A, y)

alpha = 0.005;
qm = size(A,1);
z = zeros(qm, 1);
while(norm(grad_lin_obj(W, y, A, z),2) >= 10^(-5))
    norm(grad_lin_obj(W, y, A, z),2)
    z = z - alpha*grad_lin_obj(W, y, A, z)'
end

val = z;

end

function mat = createA(W, x)
    mat = [];
    for c = 1 : length(W)
        if W(c) == 1
            mat = [mat;g1_grad(x)];
        elseif W(c) == 2
            mat = [mat;g2_grad(x)];
        elseif W(c) == 3
            mat = [mat;g3_grad(x)];
        elseif W(c) == 4
            mat = [mat;g4_grad(x)];
        end       
    end
    mat = [mat;[2*x(1),2*x(2),2*x(3),2*x(4)]];
end


function val = objective(x)

Q = [2,1,0,10;1,4,3,0.5;0,3,-5,6;10,0.5,6,-7];
b = [-1;0;-2;3];

val = (1/2)*x'*Q*x + b'*x;

end

function val = gradObj(x)

Q = [2,1,0,10;1,4,3,0.5;0,3,-5,6;10,0.5,6,-7];
b = [-1;0;-2;3];

val = Q*x + b;
val = val';
end

function set = findW(x, del)

set = [];
if g1(x) <= 0 && g1(x) > -del
    set = [set 1];
end
if g2(x) <= 0 && g2(x) > -del
    set = [set 2];
end
if g3(x) <= 0 && g3(x) > -del
    set = [set 3];
end
if g4(x) <= 0 && g4(x) > -del
    set = [set 4];
end

end
