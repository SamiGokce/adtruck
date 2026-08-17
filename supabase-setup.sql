-- Run this in Supabase SQL Editor (Dashboard > SQL Editor > New Query)

-- Bookings table
CREATE TABLE bookings (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  booking_date DATE NOT NULL,
  booking_type TEXT NOT NULL CHECK (booking_type IN ('full_day', 'morning', 'afternoon')),
  start_time TEXT NOT NULL,
  end_time TEXT NOT NULL,
  client_name TEXT NOT NULL,
  client_email TEXT NOT NULL,
  client_phone TEXT,
  status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'confirmed', 'cancelled')),
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Index for fast availability lookups
CREATE INDEX idx_bookings_date ON bookings (booking_date);

-- Enable Row Level Security
ALTER TABLE bookings ENABLE ROW LEVEL SECURITY;

-- Allow anyone to read bookings (to check availability)
CREATE POLICY "Anyone can view bookings"
  ON bookings FOR SELECT
  USING (true);

-- Allow anyone to insert bookings (to make a booking)
CREATE POLICY "Anyone can create bookings"
  ON bookings FOR INSERT
  WITH CHECK (true);
