function  [vsVP, vClass] = FACADE_orderVP_Mahattan(vsVP, vsEdges, vClass)
    % This function only sorts the vsVP, does not change the vsVP.
    vSupport = FACADE_get_ManhattanSupport(vsVP, vsEdges, vClass);
    [vsVP, vClass] = FACADE_sortClass(vsVP, vClass, vSupport);
    
