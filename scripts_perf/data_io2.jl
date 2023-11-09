function write_h5(path,fields,dim_g,I,args...)
    if !HDF5.has_parallel() && (length(args)>0)
        @warn("HDF5 has no parallel support.")
    end
    h5open(path, "w", args...) do io
        for (name,field) ∈ fields
            dset               = create_dataset(io, "/$name", datatype(eltype(field)), dataspace(dim_g))
            dset[I.indices...] = Array(field)
        end
    end
    return
end

function write_xdmf(path,h5_names,fields,dim_g,timesteps)
    xdoc = XMLDocument()
    xroot = create_root(xdoc, "Xdmf")
    set_attribute(xroot, "Version","3.0")

    xdomain     = new_child(xroot, "Domain")
    xcollection = new_child(xdomain, "Grid")
    set_attribute(xcollection, "GridType","Collection")
    set_attribute(xcollection, "CollectionType","Temporal")

    for (it,tt) ∈ enumerate(timesteps)
        h5_path = h5_names[it]

        xgrid   = new_child(xcollection, "Grid")
        set_attribute(xgrid, "GridType","Uniform")
        xtopo = new_child(xgrid, "Topology")
        set_attribute(xtopo, "TopologyType", "3DSMesh")
        set_attribute(xtopo, "Dimensions", join(reverse(dim_g),' '))

        xtime = new_child(xgrid, "Time")
        set_attribute(xtime, "Value", "$tt")

        xgeom = new_child(xgrid, "Geometry")
        set_attribute(xgeom, "GeometryType", "X_Y_Z")

        xgx = new_child(xgeom, "DataItem")
        set_attribute(xgx, "Name", "X")
        set_attribute(xgx, "Format", "HDF")
        set_attribute(xgx, "NumberType", "Float")
        set_attribute(xgx, "Precision", "8")
        set_attribute(xgx, "Dimensions", join(reverse(dim_g), ' '))
        add_text(xgx, "$(h5_path):/X")

        xgy = new_child(xgeom, "DataItem")
        set_attribute(xgy, "Name", "Y")
        set_attribute(xgy, "Format", "HDF")
        set_attribute(xgy, "NumberType", "Float")
        set_attribute(xgy, "Precision", "8")
        set_attribute(xgy, "Dimensions", join(reverse(dim_g), ' '))
        add_text(xgy, "$(h5_path):/Y")

        xgz = new_child(xgeom, "DataItem")
        set_attribute(xgz, "Name", "Z")
        set_attribute(xgz, "Format", "HDF")
        set_attribute(xgz, "NumberType", "Float")
        set_attribute(xgz, "Precision", "8")
        set_attribute(xgz, "Dimensions", join(reverse(dim_g), ' '))
        add_text(xgz, "$(h5_path):/Z")

        fields2 = filter(tuple -> first(tuple) != "X" && first(tuple) != "Y" && first(tuple) != "Z", fields)
        for (name,field) ∈ fields2
            create_xdmf_attribute(xgrid,h5_path,name,size(field))
        end
    end

    save_file(xdoc, path)
    return
end

function write_xdmf(path,h5_path,fields,dim_g)
    xdoc = XMLDocument()
    xroot = create_root(xdoc, "Xdmf")
    set_attribute(xroot, "Version","3.0")

    xdomain = new_child(xroot, "Domain")
    xgrid   = new_child(xdomain, "Grid")
    set_attribute(xgrid, "GridType","Uniform")
    xtopo = new_child(xgrid, "Topology")
    set_attribute(xtopo, "TopologyType", "3DSMesh")
    set_attribute(xtopo, "Dimensions", join(reverse(dim_g),' '))

    xgeom = new_child(xgrid, "Geometry")
    set_attribute(xgeom, "GeometryType", "X_Y_Z")

    xgx = new_child(xgeom, "DataItem")
    set_attribute(xgx, "Name", "X")
    set_attribute(xgx, "Format", "HDF")
    set_attribute(xgx, "NumberType", "Float")
    set_attribute(xgx, "Precision", "8")
    set_attribute(xgx, "Dimensions", join(reverse(dim_g), ' '))
    add_text(xgx, "$(h5_path):/X")

    xgy = new_child(xgeom, "DataItem")
    set_attribute(xgy, "Name", "Y")
    set_attribute(xgy, "Format", "HDF")
    set_attribute(xgy, "NumberType", "Float")
    set_attribute(xgy, "Precision", "8")
    set_attribute(xgy, "Dimensions", join(reverse(dim_g), ' '))
    add_text(xgy, "$(h5_path):/Y")

    xgz = new_child(xgeom, "DataItem")
    set_attribute(xgz, "Name", "Z")
    set_attribute(xgz, "Format", "HDF")
    set_attribute(xgz, "NumberType", "Float")
    set_attribute(xgz, "Precision", "8")
    set_attribute(xgz, "Dimensions", join(reverse(dim_g), ' '))
    add_text(xgz, "$(h5_path):/Z")

    fields2 = filter(tuple -> first(tuple) != "X" && first(tuple) != "Y" && first(tuple) != "Z", fields)
    for (name,field) ∈ fields2
        create_xdmf_attribute(xgrid,h5_path,name,size(field))
    end
    save_file(xdoc, path)
    return
end

function create_xdmf_attribute(xgrid,h5_path,name,dim_g)
    # TODO: solve type and precision
    xattr = new_child(xgrid, "Attribute")
    set_attribute(xattr, "Name", name)
    set_attribute(xattr, "Center", "Cell")
    xdata = new_child(xattr, "DataItem")
    set_attribute(xdata, "Format", "HDF")
    set_attribute(xdata, "NumberType", "Float")
    set_attribute(xdata, "Precision", "8")
    set_attribute(xdata, "Dimensions", join(reverse(dim_g), ' '))
    add_text(xdata, "$(h5_path):/$name")
    return xattr
end
