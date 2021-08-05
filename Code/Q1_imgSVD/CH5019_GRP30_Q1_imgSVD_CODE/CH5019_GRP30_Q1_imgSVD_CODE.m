%--------------------------------IMG_COMP_MAIN_FUNCTION----------------------------------------------------------------
% we can put our input variables(no of
%pixels,subjects,and pics per subject) along with the source address to
%collect data
x = 4096;
y = 10;
z = 15;
full_img = pull_data('C:\Users\Avinash\Desktop\Term project 2020\Dataset_Question1\',x,y,z); % remember to  change 
                                                                                              % the address acc to ur system

%the above given statement creates a 
comp = 6; %--> number of terms to be included in the SVD
rep_img = create_rep_img(full_img,x,y,z,comp);

count=0;
result = zeros(z,y);
i=1;
while(i<=z)   %first loop iterates over the list if subjects
    
    j=1;
    while(j<=y)  %the first inner loop iterates over the images of subject
        temper = full_img(:,j,i);
        q = 1;
        normval = zeros(z,0);
        while(q<=z)         %this iterates over the subjects rep img
            temper = double(temper);
            w = double(rep_img(:,q));
            normval(q) = norm(temper'-w');   %calculates the norm of the diff b/w rep and original img
            
            q=q+1;
        end
        [MIN,INDEX]=min(normval);   %finds the min norm and the index of the corresponding subject
        
        if (INDEX==i) 
            count=count+1;    %increments count if identified correctly
            result(i,j)=1;
        else
            result(i,j)=0;    % the result matrix allows us to identify which
                              % img was identified incorrectly( 1 if
                              % correct 0 if wrong)
        end    
        
        j=j+1;
    end
    i=i+1;
end
count      %count tells the number of correct identifications of subject

%----------------------------------PULL_DATA_FUNCTION------------------------------------------------------

function full_img = pull_data(main_address,x,y,z)
full_img = zeros(x,y,z);
full_img = uint8(full_img);   
i = 1;

while(i<=z)
   
   
   m = strcat(main_address,int2str(i),'\') %adds folder no. to the address
   
   j=1;
   s=zeros(x,y);
   s=uint8(s);
   while(j<=y)
       
       n = strcat(m,int2str(j),'.pgm') ;  %creates final address for img to be loaded
       p = importdata(n);
       imshow(p);
       p = reshape(p,[4096,1]);   %reshapes img into vector
       s(:,j)=p;
       
       j=j+1;
   end    
   full_img(:,:,i)=s ;   %loads img into the main uncompressed data matrix
   i = i+1; 
  
end
end




%---------------------------------CREATE_REP_IMG_FUNCTION----------------------------------------------------

function rep_img = create_rep_img(full_img,x,y,z,comp)

rep_img = zeros(x,z)
rep_img = uint8(rep_img)
i = 1;
while(i<=z)  % this loop creates avg img for each subject
    
    rep = 0
    j=1;
    while(j<=y)
        t = full_img(:,j,i);
        t = t/10;
        rep = rep + t;
        
        
        j=j+1;
    end
    %figure;
    %imshow(reshape(rep1,[64,64])) %--> uncomment to view the mean image
    
    rep = reshape(rep,[64,64]);  %reshape for SVD
    rep = double(rep);
    [U,S,V] = svd(rep);
    f = comp;  %--> f denotes the number of singular values components we take
    k = 0;
    while(f>0)    %loop keeps adding more SVD layers on each iteration
        col = U(:,f);
        col2 = V(:,f);
        sval = S(f,f);
        temp = col*col2';
        temp = temp*sval;
        k = k+temp;
        
        
        f=f-1;
    end
    k = uint8(k);
    
    figure;
    imshow(k);
    rep_img(:,i) = reshape(k,[4096,1]);   %loads rep img into the compressed data matrix

    i=i+1;
end
end
