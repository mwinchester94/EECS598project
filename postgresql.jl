using LibPQ, DataStreams, DataFrames

function check_postgresql_files(connection_dir::String)
    if !isfile(joinpath(connection_dir, "postgresql.crt"))
        error("postgresql.crt missing from $(connection_dir)")
    elseif !isfile(joinpath(connection_dir, "postgresql.key"))
        error("postgresql.key missing from $(connection_dir)")
    elseif !isfile(joinpath(connection_dir, "root.crt"))
        error("root.crt missing rom $(connection_dir)")
    end
    return
end

function connect_missing_pitch(connection_dir::String, username::String, password::String)
    check_postgresql_files(connection_dir)
    host = "host=141.211.55.211"
    port = "port=58420"
    user = "user=$(username)"
    password = "password=$(password)"
    sslcert = "sslcert="*joinpath(connection_dir, "postgresql.crt")
    sslkey = "sslkey="*joinpath(connection_dir, "postgresql.key")
    sslrootcert = "sslrootcert="*joinpath(connection_dir, "root.crt")
    sslmode = "sslmode=verify-ca"
    dbname = "dbname=missing-pitch"

    # this connection string tells libpq where to look, and how to
    # authenticate with the database
    conn_string = join([
            host,
            port,
            user,
            password,
            sslcert,
            sslkey,
            sslrootcert,
            sslmode,
            dbname
            ], " ")

    println("Connecting to missing-pitch database.")
    println("Data are stored in the 'pitchmissing' table.")

    # connect to database
    return LibPQ.Connection(conn_string)
end

function connect_probability_out(connection_dir::String, username::String, password::String)
    check_postgresql_files(connection_dir)
    host = "host=141.211.55.211"
    port = "port=58422"
    user = "user=$(username)"
    password = "password=$(password)"
    sslcert = "sslcert="*joinpath(connection_dir, "postgresql.crt")
    sslkey = "sslkey="*joinpath(connection_dir, "postgresql.key")
    sslrootcert = "sslrootcert="*joinpath(connection_dir, "root.crt")
    sslmode = "sslmode=verify-ca"
    dbname = "dbname=probability-out"

    # this connection string tells libpq where to look, and how to
    # authenticate with the database
    conn_string = join([
            host,
            port,
            user,
            password,
            sslcert,
            sslkey,
            sslrootcert,
            sslmode,
            dbname
            ], " ")

    println("Connecting to probability-out database.")
    println("Data are stored in 'statcasttest' and 'statcasttrain' tables.")

    # connect to the database
    return LibPQ.Connection(conn_string)
end

function connect_pitch_tunneling(connection_dir::String, username::String, password::String)
    check_postgresql_files(connection_dir)
    host = "host=141.211.55.211"
    port = "port=58421"
    user = "user=$(username)"
    password = "password=$(password)"
    sslcert = "sslcert="*joinpath(connection_dir, "postgresql.crt")
    sslkey = "sslkey="*joinpath(connection_dir, "postgresql.key")
    sslrootcert = "sslrootcert="*joinpath(connection_dir, "root.crt")
    sslmode = "sslmode=verify-ca"
    dbname = "dbname=pitch-tunneling"

    # this connection string tells libpq where to look, and how to
    # authenticate with the database
    conn_string = join([
            host,
            port,
            user,
            password,
            sslcert,
            sslkey,
            sslrootcert,
            sslmode,
            dbname
            ], " ")

    println("Connecting to pitch-tunneling database.")
    println("Data are stored in 'pitchtunnelingtest' and 'pitchtunnelingtrain' tables.")

    # connect to the database
    return LibPQ.Connection(conn_string)
end

function stream_to_dataframe(conn::LibPQ.Connection, query::String)
    return DataFrame(Data.stream!(execute(conn, query), Data.Table))
end
