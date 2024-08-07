import gmpy2
import numpy as np
import random
import functools
import torch
import math
import tqdm
import bplib as bp
import base64
import hashlib
from Crypto import Random
from Crypto.Cipher import AES
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import dh
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.asymmetric import dsa

_RINT = functools.partial(random.SystemRandom().randint, 0)

def identitymatrix(n):
    return [[int(x == y) for x in range(n)] for y in range(n)]

def inversematrix(matrix, q=285191):
    """
    function to compute modular inverse of matrix
    """
    n = len(matrix)
    A = np.matrix([[matrix[j, i] for i in range(n)] for j in range(n)], dtype = int)
    A = A % q
    Ainv = np.matrix(identitymatrix(n), dtype = int)
    for i in range(n):
        factor = int(gmpy2.invert(gmpy2.mpz(A[i,i]), gmpy2.mpz(q)))
        if factor is None:
             raise ValueError("TODO: deal with this case")
        A[i] = A[i] * factor % q
        Ainv[i] = Ainv[i] * factor % q
        for j in range(n):
            if (i != j):
                factor = A[j, i]
                A[j] = (A[j] - factor * A[i]) % q
                Ainv[j] = (Ainv[j] - factor * Ainv[i]) % q
    return Ainv

def _eval_at(poly, x, prime):
    """Evaluates polynomial (coefficient tuple) at x, used to generate a
    shamir pool in make_random_shares below.
    """
    accum = 0
    for coeff in reversed(poly):
        accum *= x
        accum += coeff
        accum %= prime
    return accum

def make_random_shares(secret, minimum, shares, prime=285191):
    """
    Generates a random shamir pool for a given secret, returns share points.
    """
    if minimum > shares:
        raise ValueError("Pool secret would be irrecoverable.")
    poly = [secret] + [_RINT(prime - 1) for i in range(minimum - 1)]
    points = [(i, _eval_at(poly, i, prime))
              for i in range(1, shares + 1)]
    return points


def get_constant_matrix(n, t, k, share_pts, muti_secret_pts, singly_secret_pt, p=285191):
    """
    get matrix to multiply packed shared secrets
    """
    mats = []
    for i in tqdm(range(k)):
        # create the inverse of B_ei
        this_secret_pt = muti_secret_pts[i]
        B_ei = np.ones((n, n), dtype=int)
        for j in range(n):
            for l in range(n):
                B_ei[j, l] = (share_pts[l] - this_secret_pt)**j % p
        this_mat = np.array(inversematrix(B_ei, p))

        # compute the vandermonde matrix for singly secret (chopped by t)
        B_ei2 = np.zeros((n, n), dtype=int)
        B_ei2[0] = 1
        for j in range(1, t):
            for l in range(n):
                B_ei2[j, l] = (share_pts[l] - singly_secret_pt)**j % p
        this_mat = (this_mat.dot(B_ei2)) % p

        mats.append(this_mat)
    return mats

def multi_to_singly(shares, t, k, share_pts, muti_secret_pts, singly_secret_pt, p=285191):
    """
    function to convert packed secret sharing to singly shared secret
    """
    n = len(shares)
    matrices = get_constant_matrix(n, t, k, share_pts, muti_secret_pts, singly_secret_pt, p)
    multi_shares = np.array([s[1] for s in shares], dtype = int)
    singly_shares = []
    for i in range(k):
        mat = matrices[i]
        this_singly_shares = multi_shares.dot(mat) % p
        singly_shares.append(list(zip(share_pts, this_singly_shares)))
    return singly_shares

def setup_vss(t=50, p=285191):
    G = bp.BpGroup()
    # g1, g2 = G.gen1(), G.gen2()
    g1 = G.gen1()
    sk = random.randrange(p)
    pks = []
    for i in range(t+1):
        pks.append(g1.exp(sk**i))
    return sk, pks, G

def compute_denominators(points, p=285191):
    denums = []
    for i in range(len(points)):
        xi = points[i]
        denum = 1
        for j in range(len(points)):
            if j != i:
                xj = points[j]
                denum = (denum * (xj - xi)) % p
                denum = gmpy2.invert(denum, p)
        denums.append(denum)
    return denums

def compute_coeff(points, p=285191):
    deunms = compute_denominators(points, p)
    coefficients = deunms
    for i in range(len(coefficients)):
        coefficients[i] = gmpy2.invert(coefficients[i], p)
    for k in range(len((points))):
        for j in range(k, 0, -1):
            coefficients[j+1] += coefficients[j]
            coefficients[j] -= points[k]*coefficients[j]
    return coefficients

def create_commitment(poly, pks):
    d = len(poly)
    C = 1
    for pk, coeff in zip(pks[:d], poly):
        C = C * pk.exp(coeff)
    return C

def create_witness(poly, sk, pks, i, p=285191):
    phi_a = _eval_at(poly, sk, p)
    phi_i = _eval_at(poly, i, p)
    phi_i_a = (phi_a - phi_i) * gmpy2.invert(sk - i, p) % p
    g = pks[0]
    w = g.exp(phi_i_a)
    return w

def verify_vss(pks, C, i, phi_i, w, G):
    lhs = G.pair(C, pks[0])
    rhs = G.pair(w, pks[1]/pks[0].exp(i))
    rhs = rhs * (G.pair(pks[0], pks[0])).exp(phi_i)
    return rhs == lhs

def _lagrange_constants_for_point(points, point, p=285191, denums=None):
    constants = [0] * len(points)
    for i in range(len(points)):
        xi = points[i]
        num = 1
        if denums is None:
            denum = 1
        for j in range(len(points)):
            if j != i:
                xj = points[j]
                num = (num * (xj - point)) % p
                if denums is None:
                    denum = (denum * (xj - xi)) % p
        if denums is None:
            denum = gmpy2.invert(denum, p)
        else:
            denum = denums[i]
        constants[i] = int((num * denum) % p)
    return constants

def _lagrange_interpolate(x, x_s, y_s, p=285191, denums=None):
    """
    Find the y-value for the given x, given n (x, y) points;
    k points will define a polynomial of up to kth order.
    """
    k = len(x_s)
    assert k == len(set(x_s)), "points must be distinct"
    constants = _lagrange_constants_for_point(x_s, x, p, denums)
    out = 0
    for ci, vi in zip(constants, y_s):
        out += ((ci * vi) % p)
        out %= p
    return out
    # return sum(ci * vi for ci, vi in zip(constants, y_s)) % p

def recover_singly_secret(shares, secret_pt, prime=285191, denums=None):
    """
    Recover the secret from share points
    (points (x,y) on the polynomial).
    """
    if len(shares) < 3:
        raise ValueError("need at least three shares")
    x_s, y_s = zip(*shares)
    return _lagrange_interpolate(secret_pt, x_s, y_s, prime, denums)

def sample_packed_polynomial(secrets, secret_pts, rand_pts, minimum, k, p=285191):
    """
    generate random points for packed shares
    """
    points = secret_pts + rand_pts
    values = secrets + [_RINT(p-1) for _ in range(minimum-k)]
    return (points, values)

def packed_share(secrets, secret_pts, rand_pts, minimum, k, share_pts, p=285191, denums=None):
    """
    generate random packed shares
    """
    points, values = sample_packed_polynomial(secrets, secret_pts, rand_pts, minimum, k, p)
    shares = [(s, _lagrange_interpolate(s, points, values, p, denums)) for s in share_pts]
    return shares

def packed_share_vss(secrets, secret_pts, rand_pts, minimum, k, share_pts, sk, pks, G, p=285191, denums=None):
    """
    generate random packed shares
    """
    points, values = sample_packed_polynomial(secrets, secret_pts, rand_pts, minimum, k, p)
    shares = [(s, _lagrange_interpolate(s, points, values, p, denums)) for s in share_pts]
    poly = compute_coeff(points, p)
    C = create_commitment(poly, pks)
    ws = []
    for share in shares:
        ws.append(create_witness(poly, sk, pks, share[0], p=285191))
    return shares, C, ws

def recover_packed_secret(shares, secret_pts, p=285191, denums=None):
    points, values = zip(*shares)
    secrets = [_lagrange_interpolate(s, points, values, p, denums) for s in secret_pts]
    return secrets

def dh_exchange():
    parameters = dh.generate_parameters(generator=2, key_size=2048)
    server_private_key = parameters.generate_private_key()
    peer_private_key = parameters.generate_private_key()
    shared_key = server_private_key.exchange(peer_private_key.public_key())
    # Perform key derivation.
    derived_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b'handshake data',
    ).derive(shared_key)
    same_shared_key = peer_private_key.exchange(
        server_private_key.public_key()
    )
    same_derived_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b'handshake data',
    ).derive(same_shared_key)
    assert derived_key == same_derived_key
    return same_derived_key

class AESCipher(object):

    def __init__(self, key):
        self.bs = AES.block_size
        self.key = hashlib.sha256(key.encode()).digest()

    def encrypt(self, raw):
        raw = self._pad(raw)
        iv = Random.new().read(AES.block_size)
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return base64.b64encode(iv + cipher.encrypt(raw.encode()))

    def decrypt(self, enc):
        enc = base64.b64decode(enc)
        iv = enc[:AES.block_size]
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return AESCipher._unpad(cipher.decrypt(enc[AES.block_size:])).decode('utf-8')

    def _pad(self, s):
        return s + (self.bs - len(s) % self.bs) * chr(self.bs - len(s) % self.bs)

    @staticmethod
    def _unpad(s):
        return s[:-ord(s[len(s)-1:])]

class DSASignature(object):

    def __init__(self, key=None):
        if key is None:
            key = dsa.generate_private_key(key_size=1024)
        self.key = key
        self.public_key = self.key.public_key()

    def sign(self, data):
        signature = self.key.sign(data, hashes.SHA256())
        return signature

    def verify(self, data, signature):
        self.public_key.verify(signature, data, hashes.SHA256())

def gaussian_noise(data_shape, args):
    sigma = np.sqrt(2 * np.log(1.25 / args.delta)) / args.epsilon
    return torch.normal(0, sigma*args.norm_clip, data_shape).to(args.device)