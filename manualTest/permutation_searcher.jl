
#=
foreach(p1 -> begin
    foreach(p2 -> begin
        
        global y
        global expected_qkv_x_reshaped
        global expected_qkv_x_reshaped_perm
        global B, H, W, self_num_heads

        y_copy = copy(y)

        y_copy2 = permutedims(reshape(
            permutedims(y_copy, p1),
            B,
            H * W,
            3,
            self_num_heads,
            :
            ), p2)
        
        
        if size(y_copy2) == size(expected_qkv_x_reshaped) 
            if isapprox(y_copy2, expected_qkv_x_reshaped, rtol=1e-4)
                println(
                    "y resh corretto con permutazioni: ",
                    p1,
                    "\t",
                    p2,
                    )
            end
        end

        if size(y_copy2) == size(expected_qkv_x_reshaped_perm) 
            if isapprox(y_copy2, expected_qkv_x_reshaped_perm, rtol=1e-4)
                println(
                    "y perm corretto con permutazioni: ",
                    p1,
                    "\t",
                    p2,
                    )
            end
        end
                
    end, permutations(1:5))
end, permutations(1:4))
=#