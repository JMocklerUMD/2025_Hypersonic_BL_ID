function grad_dirn = get_grad_dirn(seg_dirn)

% Returns the direction of the normal to the curve (assumed to be the
% gradient direction) - possible directions are nn, ne, ee, se
% checked - ok

if strcmp(seg_dirn,'nn') | strcmp(seg_dirn,'ss')
  grad_dirn = 'ee';
elseif strcmp(seg_dirn,'ne') | strcmp(seg_dirn,'sw')
  grad_dirn = 'se';
elseif strcmp(seg_dirn,'ee') | strcmp(seg_dirn,'ww')
  grad_dirn = 'nn';
else
  grad_dirn = 'ne';
end