function [layers, grad] = bprop(layers)
len = length(layers)    ;
    for l=len:-1:2
        if (layers{l-1}.type == 'h')
                d = layers{l}.delta;
                w = layers{l}.W;
                z = layers{l-1}.output;
                grad_a = 1 - z.^2;
                layers{l-1}.delta = grad_a .* (w'*d);
                output = layers{l-1}.output;
                layers{l}.grad_e = d * output';
                grad= sum(sum(layers{l}.grad_e));
                
        else 
            a = layers{l-1}.output;
            d = layers{l}.delta;
            layers{l}.grad_e = d * a';
            grad= sum(sum(layers{l}.grad_e));
        end
       
        
    end
end


