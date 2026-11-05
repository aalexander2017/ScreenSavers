CC      ?= gcc
CFLAGS  ?= -O2 -Wall
LDLIBS   = -lGL -lX11 -lm
PREFIX  ?= $(HOME)/.local
CONFDIR ?= /usr/share/xscreensaver/config

HACKS = cosmogenesis/cosmogenesis nebulae/nebulae

all: $(HACKS)

cosmogenesis/cosmogenesis: cosmogenesis/cosmogenesis.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

nebulae/nebulae: nebulae/nebulae.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

# binaries go to ~/.local/bin; the settings panels need root
install: all
	install -d $(PREFIX)/bin
	install -m 755 $(HACKS) $(PREFIX)/bin/
	sudo install -m 644 cosmogenesis/cosmogenesis.xml nebulae/nebulae.xml $(CONFDIR)/

clean:
	rm -f $(HACKS)

.PHONY: all install clean
