function [J, codestr] = gen_JACcode(v, q, filename, comment)
% Erhan Oztop June 2021
% Create a Jacobian matrix code for delv/delq  (v is a length column vector) 
if exist('filename','var')       %  filename can be ../xx/yy/zzz/abc.m
    dot = strfind(filename,'.');
    slash = strfind(filename,'/');  % be careful in windows folders
    dot = dot(end);     % last dot   xxxx.m
    if isempty(slash), slash = 1; else, slash = slash(end); end
    functionname = filename(slash+1:dot-1);
else
    functionname = 'compute_jac';
end

m = length(v);
n = length(q);
syms x;
for i = 1:m
    for j = 1:n
        J(i,j) = x;
    end
end



for i = 1:m
    for j = 1:n
        J(i,j) = diff(v(i), q(j));
    end
end

args='';
for j=1:n
    if (j<n)
        args = sprintf('%s%s, ',args,char(q(j)));
    else
        args = sprintf('%s%s',args,char(q(j)));
    end
end

if ~exist('comment','var'), comment = ' Analytic Jacobian'; end
codestr=sprintf('%% %s\n',comment);

codestr = sprintf('%sfunction J = %s(%s)\n',codestr,functionname,args);
sprintf('%s    J=zeros(%d,%d);\n', codestr, m, n);
for i = 1:m
    for j = 1:n
        codestr=sprintf('%s    J(%d,%d) = %s;\n',codestr,i,j,char(vpa(J(i,j))));
    end
end
codestr = sprintf('%s%%end of Jacobian computation\n', codestr);


if exist('filename','var')
    fid= fopen(filename,'w');
    if (fid>0)
        fprintf(fid, '%s\n',codestr);
        fclose(fid);
        fprintf('Jacobian code is written to %s \n', filename);
    else
       fprintf('Jacobian code file %s cannot be created \n', filename); 
    end
end