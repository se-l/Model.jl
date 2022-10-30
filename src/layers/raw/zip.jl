using Distributed
addprocs(4)

@everywhere import GZip
@everywhere import ZipFile

@everywhere function zip_file(root, file)
    fn_path = joinpath(root, file)
    if endswith(file, "zip")
        return
    elseif endswith(file, "txt.gz")
        fn_path_zip = replace(fn_path, "txt.gz" => "zip")
        fn_txt = replace(file, ".gz" => "")
        fh = GZip.open(fn_path)
        w = ZipFile.Writer(fn_path_zip)
        f = ZipFile.addfile(w, fn_txt, method=ZipFile.Deflate)

        ZipFile.write(f, read(fh, String));

        ZipFile.close(w)
        GZip.close(fh)

    elseif endswith(file, "txt")
        fn_path_zip = replace(fn_path, "txt" => "zip")
        fn_txt = file

        fh = open(fn_path)
        w = ZipFile.Writer(fn_path_zip)
        f = ZipFile.addfile(w, fn_txt, method=ZipFile.Deflate)

        ZipFile.write(f, read(fh, String));

        ZipFile.close(w)
    end
    
    println(fn_path_zip)
    rm(fn_path)
end

function zipem()
    dir =  raw"C:\repos\trade\data\crypto\bitfinex\tick"
    for (root, dirs, files) in walkdir(dir)
        pmap(file->zip_file(root, file), files)
    end
    println("Done.")
end
