module process_data

include("postgresql.jl")

function connect_db()
  connection_dir = "./pitch-tunneling"

  user, pw = open("pw_pitch_tunneling.txt") do file
    readlines(file)
  end

  username = user
  password = pw

  return connect_pitch_tunneling(connection_dir, username, password)
end

function get_data(train=true, islhp=false, islhb=false, fast=true, sorted=false)
  query = "SELECT "

  if sorted
    query *= "gamepk, inning, istop, atbatnumber, pitchnumber, "
  end

  query *= "releasepositionx, releasepositiony, releasepositionz, tunnellocationx, tunnellocationz, trajectorylocationx, trajectorylocationz, runvalue\nFROM "

  if train
    query *= "pitchtunnelingtrain\nWHERE "
  else
    query *= "pitchtunnelingtest\nWHERE "
  end

  query *= "islhp = '"*string(Int(islhp))*"' AND islhb = '"*string(Int(islhb))*"' AND "

  if sorted
    query *= "runvalue IS NOT NULL\nORDER BY gamepk, inning, istop DESC, atbatnumber, pitchnumber;"
  else
    if fast
      query *= "(pitchtype = 'FF' OR pitchtype = 'FT' OR pitchtype = 'SI') AND runvalue IS NOT NULL;"
    else
      query *= "(pitchtype != 'FF' AND pitchtype != 'FT' AND pitchtype != 'SI') AND runvalue IS NOT NULL;"
    end
  end

  conn = connect_db()
  data = stream_to_dataframe(conn, query)
  close(conn)
  return data
end

export get_data

end
